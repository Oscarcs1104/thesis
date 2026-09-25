"""Block 3 -- does the generator actually obey the requested property change?

Ask for a delta, generate, measure the delta that came out with RDKit. No learned
predictor anywhere in the loop, so nothing here can be circular: the oracle that scores
a generated molecule is the same function that labelled the corpus, and it is exact.

    python crossmodal_model/generation/eval_oracle.py \\
        --ckpt checkpoints/pairs/pairs_graph_smiles_scratch_s2025.pt \\
        --property logp --n-seeds 200 --guidance 1.0 2.0 3.0

Design, and why it is this and not something simpler:

*   One property is requested at a time; the other three are set to their null bin. The
    model was trained with independent per-property dropout precisely so this is a
    request it has seen, and it isolates the effect instead of confounding four asks.

*   Every requested bin is asked of the SAME seed molecules. The comparison is then
    within-seed: obtained delta varies only because the request varied. This is what
    the previous evaluation got wrong -- it correlated `requested = true_y + shift`
    against an estimate that already tracked `true_y` through the seed, so a model with
    zero conditioning still scored a high Pearson. Here a model that ignores the
    condition produces the same distribution for every bin and the curve is flat.

*   A null-condition arm runs alongside. It is the honest reference: if conditional and
    unconditional generations have the same delta distribution, the condition does
    nothing, whatever the headline correlation says.

*   The copy rate is reported per bin. A model that returns the seed unchanged scores
    delta = 0, which looks excellent in the middle bins and is worth nothing. Any
    result has to be read next to this number.

Outputs, under results/oracle/<run>/:
    generations.csv   one row per generated molecule, with both deltas
    summary.json      per-bin aggregates and the headline numbers
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

warnings.filterwarnings("ignore")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from torch_geometric.data import Batch  # noqa: E402

from common.property_bins import PropertyBinner  # noqa: E402
from common.repro import seed_everything  # noqa: E402
from crossmodal_model.generation.conditional_decoder import ConditionalMoleculeGenerator  # noqa: E402
from crossmodal_model.generation.pair_data import build_pair_datasets  # noqa: E402
from crossmodal_model.model.mola_hybrid import HybridMoLA  # noqa: E402
from data_pipeline.rdkit_labels import PROPERTIES  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--corpus-dir", type=str, default="data/moses")
    p.add_argument("--property", type=str, default="logp", choices=PROPERTIES,
                   help="the one property conditioned on; the rest are set to null")
    p.add_argument("--n-seeds", type=int, default=200,
                   help="seed molecules, each asked for every bin. Total generations per "
                        "guidance value is n_seeds x (num_bins + 1), the +1 being the "
                        "null-condition control")
    p.add_argument("--bins", type=int, nargs="+", default=None,
                   help="which requested bins to sweep (default: all of them)")
    p.add_argument("--guidance", type=float, nargs="+", default=[1.0],
                   help="classifier-free guidance weights. w=1 is plain conditional "
                        "sampling; if the curve does not steepen with w, the condition "
                        "is barely being used")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true",
                   help="argmax instead of sampling. Makes uniqueness meaningless (one "
                        "seed, one output) but isolates conditioning from sampling noise")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-len", type=int, default=96)
    p.add_argument("--seed", type=int, default=2025,
                   help="must match the training run's --seed, or the test split differs "
                        "and the seed molecules were trained on")
    p.add_argument("--out-dir", type=str, default="results/oracle")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=8)
    return p.parse_args()


# --------------------------------------------------------------------------------------
# the oracle
# --------------------------------------------------------------------------------------

def property_values(smiles_list: Sequence[str]) -> np.ndarray:
    """[N, 4] RDKit properties, NaN where the string is not a molecule.

    Deliberately calls the same functions as data_pipeline/rdkit_labels.py through the
    same import, so a change there cannot leave evaluation measuring something else.
    """
    from data_pipeline.rdkit_labels import label_one

    out = np.full((len(smiles_list), len(PROPERTIES)), np.nan, dtype=np.float64)
    for i, smi in enumerate(smiles_list):
        # An empty string is not a parse failure to RDKit: MolFromSmiles("") returns an
        # empty molecule whose logP, TPSA and MW are all 0.0. Left to itself that lands
        # in the CSV as a real measurement, so it is rejected before it can.
        if not smi:
            continue
        r = label_one(smi)
        if r is not None:
            out[i] = r
    return out


def canonical_and_key(smiles_list: Sequence[str]):
    """(canonical SMILES or None, InChIKey or None) per input."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    canon: List[Optional[str]] = []
    keys: List[Optional[str]] = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            canon.append(None)
            keys.append(None)
            continue
        canon.append(Chem.MolToSmiles(mol))
        try:
            keys.append(Chem.MolToInchiKey(mol))
        except Exception:
            keys.append(None)
    return canon, keys


def tanimoto(seed_smiles: Sequence[str], gen_smiles: Sequence[Optional[str]]) -> np.ndarray:
    """ECFP4 similarity of each generation to its seed. NaN for invalid generations."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    out = np.full(len(gen_smiles), np.nan)
    cache: Dict[str, object] = {}

    def fp(smi: str):
        if smi not in cache:
            mol = Chem.MolFromSmiles(smi)
            cache[smi] = None if mol is None else AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return cache[smi]

    for i, (a, b) in enumerate(zip(seed_smiles, gen_smiles)):
        if not b:
            continue
        fa, fb = fp(a), fp(b)
        if fa is not None and fb is not None:
            out[i] = DataStructs.TanimotoSimilarity(fa, fb)
    return out


def internal_diversity(smiles: Sequence[str], p: int = 1, max_n: int = 3000,
                      seed: int = 0) -> float:
    """MOSES's IntDiv_p over a set of molecules. Higher is more varied.

        IntDiv_p(G) = 1 - mean_i ( mean_j T(m_i, m_j)^p ) ^ (1/p)

    T is the Tanimoto between ECFP4 fingerprints, and the diagonal is included, as in
    the reference implementation -- excluding it would make the metric depend on set
    size. p=2 punishes the presence of near-duplicate pairs harder than p=1.

    It complements the two diversity numbers already reported without replacing either.
    Uniqueness counts how many outputs are distinct and calls two molecules differing by
    one methyl entirely different; novelty asks whether an output is in the training
    corpus and says nothing about the outputs' relation to each other. IntDiv measures
    how far apart they actually are.

    Quadratic in the number of molecules, so a sample is taken above max_n: at 3000 this
    is nine million pairs, seconds with RDKit's bulk similarity, and the estimate is
    already stable.
    """
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    mols = [m for m in (Chem.MolFromSmiles(str(x)) for x in smiles if x) if m is not None]
    if len(mols) < 2:
        return float("nan")
    if len(mols) > max_n:
        idx = np.random.default_rng(seed).choice(len(mols), max_n, replace=False)
        mols = [mols[i] for i in idx]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]

    per_molecule = np.empty(len(fps))
    for i, fp in enumerate(fps):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, fps))
        per_molecule[i] = (sims ** p).mean() ** (1.0 / p)
    return float(1.0 - per_molecule.mean())


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------

def load_generator(ckpt_path: Path, device: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    vocab = ck["vocab"]
    vocab["id_to_token"] = {int(k): v for k, v in vocab["id_to_token"].items()}
    char_vocab = ck["char_vocab"]
    binners = {k: PropertyBinner.from_state_dict(v) for k, v in ck["binners"].items()}
    # Written explicitly by the trainer because --init-encoder overrides the CLI values.
    # Older checkpoints predate that and fall back to args, which is right for them.
    hidden_dim = int(ck.get("hidden_dim", a.get("hidden_dim", 256)))
    num_layers = int(ck.get("num_layers", a.get("num_layers", 3)))
    max_sm_len = int(a.get("max_sm_len", 100))
    gin_mult = int(ck.get("gin_hidden_mult", a.get("gin_hidden_mult", 1)))
    normalizar = bool(ck.get("normalize_branches", a.get("normalize_branches", False)))

    mola = HybridMoLA(
        sm_vocab_size=len(char_vocab), hidden_dim=hidden_dim, output_dim=1,
        num_layers=num_layers, positional_smiles=True, max_sm_len=max_sm_len,
        use_graph=bool(a.get("use_graph", True)), use_smiles=bool(a.get("use_smiles", True)),
        gin_hidden_mult=gin_mult,
        normalize_branches=normalizar,
    )
    model = ConditionalMoleculeGenerator(
        mola, vocab_size=len(vocab["token_to_id"]), hidden_dim=hidden_dim,
        pad_idx=vocab["pad_idx"],
        cond_vocab_sizes=[binners[n].num_bins + 1 for n in PROPERTIES],
        cond_null_bins=[binners[n].null_bin for n in PROPERTIES],
        cond_dropout=float(a.get("cond_dropout", 0.15)),
        # From the checkpoint, not from a default: a model trained with the fusion in its
        # memory has a differently shaped memory and rebuilding it without would evaluate
        # a different architecture than the weights belong to.
        fusion_in_memory=bool(ck.get("fusion_in_memory", a.get("fusion_in_memory", False))),
        decoder_layers=int(a.get("decoder_layers", 6)),
        max_len=max_sm_len + 32,
    )
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
    if missing or unexpected:
        raise SystemExit("checkpoint did not load cleanly -- the model was rebuilt with the "
                         f"wrong shape.\n  missing:    {list(missing)[:8]}\n"
                         f"  unexpected: {list(unexpected)[:8]}")
    model.to(device).eval()
    return model, vocab, binners, ck


# --------------------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------------------

def generate_for_request(model, cache, seed_idx: np.ndarray, cond_row: List[int],
                         vocab: Dict, device: str, batch_size: int, max_len: int,
                         temperature: float, sample: bool, guidance: float) -> List[str]:
    """Same condition asked of every seed in seed_idx; returns one string per seed."""
    out: List[str] = []
    cond_template = torch.tensor(cond_row, dtype=torch.long, device=device)
    for start in range(0, len(seed_idx), batch_size):
        chunk = seed_idx[start:start + batch_size]
        batch = Batch.from_data_list([cache.get(int(i)) for i in chunk]).to(device)
        cond = cond_template.unsqueeze(0).expand(len(chunk), -1).contiguous()
        out.extend(model.generate_batch(batch, cond, vocab, max_len=max_len,
                                        temperature=temperature, sample=sample,
                                        guidance=guidance))
    return out


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, deterministic=False)
    device = args.device
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    run_name = args.run_name or f"{ckpt_path.stem}_{args.property}"
    out_dir = ROOT / args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model, vocab, binners, ck = load_generator(ckpt_path, device)
    arm = ck.get("arm", "unknown")
    prop_idx = PROPERTIES.index(args.property)
    binner = binners[args.property]
    null_row = [binners[n].null_bin for n in PROPERTIES]
    print(f"checkpoint {ckpt_path.name} | arm {arm} | step {ck.get('step')}")
    print(f"conditioning on {args.property}: {binner.num_bins} bins, null at {binner.null_bin}")

    # Same split as training: the seeds must come from molecules the model never saw.
    _, _, test_ds, cache, _, _ = build_pair_datasets(
        ROOT / args.corpus_dir, num_bins=int(ck.get("args", {}).get("num_bins", 20)),
        max_sm_len=int(ck.get("args", {}).get("max_sm_len", 100)),
        seed=args.seed, workers=args.num_workers,
    )

    # The character vocabulary is built from the corpus on disk, while the SMILES
    # embedding was sized and trained against the one in the checkpoint. If they differ,
    # a character index can exceed the embedding's rows and CUDA raises a device-side
    # assert -- asynchronously, so the traceback points at whatever kernel launched next
    # rather than at the lookup. Widening the corpus is exactly how they come to differ:
    # new molecules bring characters the old vocabulary never had.
    ck_vocab = ck.get("char_vocab")
    if ck_vocab is not None and ck_vocab != cache.char_vocab:
        only_here = sorted(set(cache.char_vocab) - set(ck_vocab))
        raise SystemExit(
            f"The corpus at {args.corpus_dir} has a different character vocabulary than "
            f"the checkpoint was trained with ({len(cache.char_vocab)} characters against "
            f"{len(ck_vocab)}).\n"
            + (f"  Only in the corpus: {only_here[:12]}\n" if only_here else "")
            + f"  Every row of the SMILES embedding would stand for a different character.\n"
            f"  Point --corpus-dir at the corpus this checkpoint was trained on."
        )

    import pandas as pd

    corpus = pd.read_csv(ROOT / args.corpus_dir / "corpus.csv")
    corpus_smiles = corpus["smiles"].astype(str).to_numpy()
    train_keys = set(corpus["inchikey"].astype(str).tolist())

    seed_pool = np.unique(test_ds.pairs[:, 0])
    rng = np.random.default_rng(args.seed)
    seed_idx = rng.choice(seed_pool, size=min(args.n_seeds, len(seed_pool)), replace=False)
    seed_idx.sort()
    seed_smiles = corpus_smiles[seed_idx].tolist()
    seed_props = property_values(seed_smiles)
    print(f"{len(seed_idx)} seed molecules from the test split "
          f"({len(seed_pool):,} available)")

    requested_bins = args.bins if args.bins is not None else list(range(binner.num_bins))
    # The null request is the control, carried through the same code path as the rest.
    requests = [("null", binner.null_bin)] + [("bin", b) for b in requested_bins]
    total = len(args.guidance) * len(requests) * len(seed_idx)
    print(f"{total:,} generations: {len(args.guidance)} guidance x {len(requests)} "
          f"requests x {len(seed_idx)} seeds")

    rows: List[dict] = []
    start_time = time.time()
    for w in args.guidance:
        for kind, b in requests:
            cond_row = list(null_row)
            cond_row[prop_idx] = int(b)
            gen = generate_for_request(model, cache, seed_idx, cond_row, vocab, device,
                                       args.batch_size, args.max_len, args.temperature,
                                       not args.greedy, w)
            canon, keys = canonical_and_key(gen)
            gen_props = property_values([c or "" for c in canon])
            sim = tanimoto(seed_smiles, canon)
            lo, hi = binner.bin_edges(int(b)) if kind == "bin" else (float("nan"), float("nan"))
            centre = binner.bin_center(int(b)) if kind == "bin" else float("nan")
            for k in range(len(seed_idx)):
                rows.append({
                    "guidance": w, "request": kind, "requested_bin": int(b),
                    "requested_centre": centre, "bin_lo": lo, "bin_hi": hi,
                    "seed_row": int(seed_idx[k]), "seed_smiles": seed_smiles[k],
                    "seed_value": float(seed_props[k, prop_idx]),
                    "generated": canon[k] or "", "valid": canon[k] is not None,
                    "generated_value": float(gen_props[k, prop_idx]),
                    "obtained_delta": float(gen_props[k, prop_idx] - seed_props[k, prop_idx]),
                    "tanimoto_to_seed": float(sim[k]),
                    "is_copy": bool(canon[k] is not None and canon[k] == seed_smiles[k]),
                    "novel": bool(keys[k] is not None and keys[k] not in train_keys),
                })
            done = len(rows)
            print(f"  w={w} {kind} {b:>3}  {done:>6,}/{total:,}  "
                  f"({time.time() - start_time:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "generations.csv", index=False)
    summary = summarize(df, binner, arm, ckpt_path, args)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report(summary)
    print(f"\nWrote {out_dir / 'generations.csv'} and {out_dir / 'summary.json'}")


def summarize(df, binner: PropertyBinner, arm: str, ckpt_path: Path, args) -> dict:
    """Per-bin aggregates plus the three numbers the chapter actually reports."""
    import pandas as pd

    out = {"checkpoint": str(ckpt_path), "arm": arm, "property": args.property,
           "n_seeds": int(df["seed_row"].nunique()), "guidance": {}}

    for w, gw in df.groupby("guidance"):
        valid = gw[gw["valid"]]
        conditioned = valid[valid["request"] == "bin"]
        null = valid[valid["request"] == "null"]

        per_bin = []
        for b, gb in conditioned.groupby("requested_bin"):
            d = gb["obtained_delta"].to_numpy()
            d = d[np.isfinite(d)]
            in_bin = ((d > gb["bin_lo"].iloc[0]) & (d <= gb["bin_hi"].iloc[0])).mean() if len(d) else float("nan")
            per_bin.append({
                "bin": int(b),
                "requested_centre": float(gb["requested_centre"].iloc[0]),
                "n_valid": int(len(d)),
                "obtained_mean": float(np.mean(d)) if len(d) else float("nan"),
                "obtained_median": float(np.median(d)) if len(d) else float("nan"),
                "obtained_std": float(np.std(d)) if len(d) else float("nan"),
                "hit_rate": float(in_bin),
                "copy_rate": float(gb["is_copy"].mean()),
                "mean_tanimoto": float(gb["tanimoto_to_seed"].mean()),
            })

        # Headline 1: does obtained delta track the request, within seed? Spearman over
        # every (seed, bin) pair. A model that ignores the condition gives ~0 here no
        # matter how good its reconstruction loss was.
        ok = conditioned[np.isfinite(conditioned["obtained_delta"])]
        rho, mae, slope = float("nan"), float("nan"), float("nan")
        constant = False
        if len(ok) > 2:
            from scipy.stats import spearmanr

            d = ok["obtained_delta"].to_numpy()
            # A model that returns the seed unchanged has a constant delta, and Spearman
            # is undefined on it. Undefined is not the useful answer -- an output that
            # does not move has no relationship with a request that does -- so it is
            # reported as zero, with a flag saying why, rather than as a NaN sitting next
            # to a copy rate the reader has to join up themselves.
            constant = bool(np.ptp(d) == 0)
            rho = 0.0 if constant else float(spearmanr(ok["requested_bin"], d).statistic)
            mae = float(np.mean(np.abs(d - ok["requested_centre"])))
            slope = float(np.polyfit(ok["requested_centre"], d, 1)[0])

        # Headline 2: the null control. Same seeds, no request.
        nd = null["obtained_delta"].to_numpy()
        nd = nd[np.isfinite(nd)]

        gvalid = gw["valid"].mean()
        uniq = valid["generated"].nunique() / max(len(valid), 1)
        # Two readings, because they answer different questions for a conditional model.
        # Globally, over every request at this guidance, diversity is inflated by the
        # requests themselves: bin 0 and bin 19 produce different molecules by design.
        # Within a bin -- 100 seeds all asked for the same delta -- it measures what is
        # actually wanted, how varied the answers to one request are.
        gen_all = valid[valid["request"] == "bin"]["generated"].tolist()
        div = {f"intdiv{q}": internal_diversity(gen_all, p=q) for q in (1, 2)}
        per_bin_div = {q: [] for q in (1, 2)}
        for _, gb in conditioned.groupby("requested_bin"):
            g = gb[gb["valid"]]["generated"].tolist()
            for q in (1, 2):
                v = internal_diversity(g, p=q)
                if np.isfinite(v):
                    per_bin_div[q].append(v)
        for q in (1, 2):
            div[f"intdiv{q}_within_bin"] = (float(np.mean(per_bin_div[q]))
                                            if per_bin_div[q] else float("nan"))
        out["guidance"][str(w)] = {
            "spearman_request_vs_obtained": rho,
            "obtained_delta_is_constant": constant,
            "mae_vs_bin_centre": mae,
            "slope_obtained_per_requested": slope,
            "validity": float(gvalid),
            "uniqueness": float(uniq),
            **div,
            "novelty": float(valid["novel"].mean()) if len(valid) else float("nan"),
            "copy_rate": float(valid["is_copy"].mean()) if len(valid) else float("nan"),
            "mean_tanimoto_to_seed": float(valid["tanimoto_to_seed"].mean()) if len(valid) else float("nan"),
            "null_control": {
                "n": int(len(nd)),
                "mean_delta": float(np.mean(nd)) if len(nd) else float("nan"),
                "std_delta": float(np.std(nd)) if len(nd) else float("nan"),
            },
            "per_bin": per_bin,
        }
    return out


def report(summary: dict) -> None:
    print(f"\n{'=' * 78}")
    print(f"{summary['arm']} | conditioning on {summary['property']} | "
          f"{summary['n_seeds']} seeds")
    print("=" * 78)
    for w, g in summary["guidance"].items():
        print(f"\nguidance w = {w}")
        flag = "  (the output never moved: it copies)" if g.get("obtained_delta_is_constant") else ""
        print(f"  spearman(requested bin, obtained delta) : {g['spearman_request_vs_obtained']:+.3f}"
              f"   <- the number that answers the question{flag}")
        print(f"  slope (obtained per unit requested)     : {g['slope_obtained_per_requested']:+.3f}"
              "   <- 1.0 would be perfect obedience")
        print(f"  MAE vs bin centre                       : {g['mae_vs_bin_centre']:.3f}")
        print(f"  validity / uniqueness / novelty         : {g['validity']:.3f} / "
              f"{g['uniqueness']:.3f} / {g['novelty']:.3f}")
        print(f"  IntDiv1 / IntDiv2  (global)             : "
              f"{g.get('intdiv1', float('nan')):.3f} / {g.get('intdiv2', float('nan')):.3f}")
        print(f"  IntDiv1 / IntDiv2  (dentro de un bin)   : "
              f"{g.get('intdiv1_within_bin', float('nan')):.3f} / "
              f"{g.get('intdiv2_within_bin', float('nan')):.3f}"
              "   <- respuestas a una MISMA peticion")
        print(f"  copy rate / mean Tanimoto to seed       : {g['copy_rate']:.3f} / "
              f"{g['mean_tanimoto_to_seed']:.3f}")
        nc = g["null_control"]
        print(f"  null control delta                      : {nc['mean_delta']:+.3f} "
              f"+/- {nc['std_delta']:.3f}")
        print("\n  Per bin. The first and last bins are open intervals, so their 'asked'"
              "\n  centre is an extrapolation and their MAE is not comparable to the middle"
              "\n  ones -- read the hit rate there instead. Middle bins are narrow by"
              "\n  construction (20 quantiles of the delta distribution), so a low hit rate"
              "\n  there is bin width as much as it is the model.")
        print(f"\n  {'bin':>4} {'asked':>9} {'got (mean)':>11} {'sd':>7} {'hit':>6} "
              f"{'copy':>6} {'tanimoto':>9}")
        for r in g["per_bin"]:
            print(f"  {r['bin']:>4} {r['requested_centre']:>+9.3f} {r['obtained_mean']:>+11.3f} "
                  f"{r['obtained_std']:>7.3f} {r['hit_rate']:>6.3f} {r['copy_rate']:>6.3f} "
                  f"{r['mean_tanimoto']:>9.3f}")


if __name__ == "__main__":
    main()
