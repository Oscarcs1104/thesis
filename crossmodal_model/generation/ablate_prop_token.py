"""Bloque 0 -- ¿usa el decoder el token de propiedad, o lo ignora?

Toma un checkpoint ya entrenado de `train_hybrid.py` y mide la CE/NLL de validación
sustituyendo el `prop_token` por valores que no llevan información sobre la molécula:

  baseline  y = f(M)                     (lo que vio entrenando)
  zeros     y = 0
  mean      y = media del split de train
  random    y ~ N(media_train, std_train)
  shuffled  y = permutación de las y verdaderas del batch  <- el test limpio: conserva
            la distribución marginal exacta, solo rompe el emparejamiento molécula-y
  dropped   sin token de propiedad en el memory

Si ΔNLL ≈ 0 en todas, el token se ignora y el "condicionamiento" no existe. El intervalo
de confianza bootstrap pareado (por molécula) es lo que hace defendible ese "≈ 0": sin él,
un ΔNLL pequeño no se distingue de ruido.

Complemento: masa de atención cross-attn sobre cada bloque del memory
(prop / nodos de grafo / caracteres SMILES), promediada sobre capas, cabezas y posiciones
target no-pad. Si la masa sobre la posición 0 ≈ 1/(1+N+L), el token está ignorado incluso
antes de mirar la pérdida. El desglose grafo-vs-SMILES es un dato aparte: dice sobre qué
rama del encoder se apoya realmente el decoder.

NOTA: deliberadamente NO se incluye el probe de "generar con memory = solo el token de
propiedad". Ese es un desplazamiento de distribución brutal para un decoder que nunca vio
un memory de longitud 1; su degradación no probaría nada sobre el condicionamiento.

Uso:
    python crossmodal_model/generation/ablate_prop_token.py \
        --checkpoint checkpoints/crossmodal/hybrid_joint/freesolv_mola_hybrid_joint_s2025.pt
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from torch_geometric.loader import DataLoader as GeomDataLoader  # noqa: E402

from crossmodal_model.data.featurize_hybrid import prepare_hybrid_data  # noqa: E402
from crossmodal_model.generation.decoder import (  # noqa: E402
    MoLAConditionalGenerator,
    build_memory,
    encode_batch,
)
from crossmodal_model.model.mola_hybrid import HybridMoLA  # noqa: E402
from crossmodal_model.train.core import DATASETS  # noqa: E402
from common.repro import seed_everything  # noqa: E402

VARIANTS = ["baseline", "zeros", "mean", "random", "shuffled", "dropped"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablación del token de propiedad en HybridMoLA generativo")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0, help="solo afecta a la variante 'random'/'shuffled'")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--limit-batches", type=int, default=None, help="smoke test")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def load_split(dataset_name: str, split: str, char_vocab: dict, max_sm_len: int):
    cfg = DATASETS[dataset_name]
    csv_path = REPO_ROOT / "data" / "deepchem_molnet" / cfg["dir"] / "csv" / f"{split}.csv"
    df = pd.read_csv(csv_path)
    smiles = df["smiles"].astype(str).tolist()
    y = df[cfg["target_col"]].astype(float).tolist()
    return prepare_hybrid_data(smiles, y, char_vocab, max_sm_len=max_sm_len)


def build_model(ckpt: dict, device: str) -> MoLAConditionalGenerator:
    cargs = ckpt["args"]
    char_vocab, selfies_vocab = ckpt["char_vocab"], ckpt["selfies_vocab"]
    mola = HybridMoLA(
        sm_vocab_size=len(char_vocab),
        hidden_dim=cargs["hidden_dim"],
        output_dim=1,
        num_layers=cargs["num_layers"],
        positional_smiles=True,
        max_sm_len=cargs["max_sm_len"],
    )
    model = MoLAConditionalGenerator(
        mola,
        vocab_size=len(selfies_vocab["token_to_id"]),
        hidden_dim=cargs["hidden_dim"],
        pad_idx=selfies_vocab["pad_idx"],
        use_property=True,
        decoder_layers=cargs["decoder_layers"],
        max_len=cargs["max_selfies_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


class AttentionRecorder:
    """Fuerza need_weights=True en el cross-attention de cada capa del decoder.

    nn.TransformerDecoderLayer._mha_block llama a self.multihead_attn(..., need_weights=False),
    así que un forward hook sobre el módulo no devuelve pesos: hay que interceptar la llamada.
    Devuelve (out, w) igual que el original, el layer solo usa [0] y no se entera.
    """

    def __init__(self, decoder: torch.nn.TransformerDecoder) -> None:
        self.mhas = [layer.multihead_attn for layer in decoder.layers]
        self._originals = []
        self.weights: list[torch.Tensor] = []

    def __enter__(self):
        for mha in self.mhas:
            orig = mha.forward
            self._originals.append(orig)

            def wrapped(*a, _orig=orig, **kw):
                kw = dict(kw)
                kw["need_weights"] = True
                kw["average_attn_weights"] = False
                out, w = _orig(*a, **kw)
                self.weights.append(w.detach())  # [B, heads, L_tgt, S_mem]
                return out, w

            mha.forward = wrapped
        return self

    def __exit__(self, *exc):
        for mha, orig in zip(self.mhas, self._originals):
            mha.forward = orig
        return False


def per_molecule_nll(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """NLL media por token, una entrada por molécula. [B, T, V] -> [B]"""
    tok = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
        reduction="none", ignore_index=pad_idx,
    ).view_as(targets)
    keep = (targets != pad_idx).float()
    return (tok * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)


def paired_bootstrap(diff: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float, float]:
    if n <= 0 or diff.size == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, diff.size, size=(n, diff.size))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, deterministic=False)
    device = args.device

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cargs = ckpt["args"]
    selfies_vocab = ckpt["selfies_vocab"]
    pad_idx = selfies_vocab["pad_idx"]
    dataset_name, max_sm_len = cargs["dataset"], cargs["max_sm_len"]
    max_selfies_len = cargs["max_selfies_len"]

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  dataset={dataset_name} hidden={cargs['hidden_dim']} layers={cargs['num_layers']} "
          f"decoder_layers={cargs['decoder_layers']}")

    model = build_model(ckpt, device)

    # Las estadísticas de las variantes 'mean'/'random' salen de TRAIN, no del split
    # evaluado: usar el split de eval sería filtrar información de él.
    train_data = load_split(dataset_name, "train", ckpt["char_vocab"], max_sm_len)
    train_y = torch.tensor([float(d.y.view(-1)[0]) for d in train_data])
    y_mean, y_std = float(train_y.mean()), float(train_y.std())
    print(f"  train y: mean={y_mean:.4f} std={y_std:.4f} (n={len(train_data)})")

    eval_data = load_split(dataset_name, args.split, ckpt["char_vocab"], max_sm_len)
    loader = GeomDataLoader(eval_data, batch_size=args.batch_size, shuffle=False)
    print(f"  eval split='{args.split}' n={len(eval_data)}\n")

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    nlls: dict[str, list[torch.Tensor]] = {v: [] for v in VARIANTS}
    accs: dict[str, list[tuple[int, int]]] = {v: [] for v in VARIANTS}
    attn_rows: list[dict] = []

    with torch.no_grad():
        for b_i, batch in enumerate(loader):
            if args.limit_batches is not None and b_i >= args.limit_batches:
                break
            batch = batch.to(device)
            dec_in, dec_tgt = encode_batch(batch.smiles, selfies_vocab, max_selfies_len, device)
            y_true = batch.y.float().view(-1, 1)
            B = y_true.size(0)

            # Un solo pase del encoder, compartido por todas las variantes: garantiza que
            # la única diferencia entre ellas es el token de propiedad.
            raw = model.mola.encode_for_generation(batch)

            prop_values = {
                "baseline": y_true,
                "zeros": torch.zeros_like(y_true),
                "mean": torch.full_like(y_true, y_mean),
                "random": (torch.randn(B, 1, generator=gen) * y_std + y_mean).to(device),
                "shuffled": y_true[torch.randperm(B, generator=gen).to(device)],
                "dropped": None,
            }

            for variant, pv in prop_values.items():
                memory, mem_pad = build_memory(
                    raw, model.hidden_dim,
                    property_values=pv,
                    property_proj=model.property_proj if pv is not None else None,
                )
                record = variant == "baseline"
                if record:
                    with AttentionRecorder(model.decoder.decoder) as rec:
                        logits = model.decoder(memory, mem_pad, dec_in)
                    attn_rows.append(summarize_attention(
                        rec.weights, mem_pad, dec_in, pad_idx, max_sm_len,
                    ))
                else:
                    logits = model.decoder(memory, mem_pad, dec_in)

                nlls[variant].append(per_molecule_nll(logits, dec_tgt, pad_idx).cpu())
                keep = dec_tgt != pad_idx
                correct = int(((logits.argmax(-1) == dec_tgt) & keep).sum())
                accs[variant].append((correct, int(keep.sum())))

    report(nlls, accs, attn_rows, args, dataset_name)


def summarize_attention(weights, mem_pad, dec_in, pad_idx, n_sm) -> dict:
    """Masa de atención por bloque del memory, promediada sobre capas/cabezas/targets."""
    tgt_keep = (dec_in != pad_idx).float()                       # [B, L_tgt]
    denom = tgt_keep.sum().clamp_min(1.0)
    n_valid = (~mem_pad).float().sum(dim=1)                      # [B] posiciones reales
    n_mem = mem_pad.size(1)
    n_nodes = n_mem - 1 - n_sm                                   # memory = [prop | nodos | chars]

    def block_mass(w, lo, hi):
        # w: [B, heads, L_tgt, S] -> masa media sobre el slice [lo, hi)
        m = w[:, :, :, lo:hi].sum(-1).mean(dim=1)                # media sobre cabezas -> [B, L_tgt]
        return float((m * tgt_keep).sum() / denom)

    prop, nodes, chars = [], [], []
    for w in weights:
        prop.append(block_mass(w, 0, 1))
        nodes.append(block_mass(w, 1, 1 + n_nodes))
        chars.append(block_mass(w, 1 + n_nodes, n_mem))
    return {
        "prop": float(np.mean(prop)),
        "nodes": float(np.mean(nodes)),
        "chars": float(np.mean(chars)),
        "uniform_ref": float((1.0 / n_valid.clamp_min(1.0)).mean()),
        "n_items": int(dec_in.size(0)),
    }


def report(nlls, accs, attn_rows, args, dataset_name) -> None:
    rng = np.random.default_rng(args.seed)
    base = torch.cat(nlls["baseline"]).numpy()
    rows = []
    for v in VARIANTS:
        arr = torch.cat(nlls[v]).numpy()
        correct = sum(c for c, _ in accs[v])
        total = sum(t for _, t in accs[v])
        diff = arr - base
        lo, hi = paired_bootstrap(diff, 0 if v == "baseline" else args.bootstrap, rng)
        rows.append({
            "variant": v,
            "nll": float(arr.mean()),
            "ppl": float(np.exp(arr.mean())),
            "delta_nll": float(diff.mean()),
            "delta_ci_lo": lo,
            "delta_ci_hi": hi,
            "token_acc": correct / max(total, 1),
        })

    print(f"{'variante':<10} {'NLL':>8} {'PPL':>8} {'ΔNLL':>9}  {'IC95% pareado':>20} {'tok_acc':>8}")
    print("-" * 70)
    for r in rows:
        ci = "" if r["variant"] == "baseline" else f"[{r['delta_ci_lo']:+.4f}, {r['delta_ci_hi']:+.4f}]"
        print(f"{r['variant']:<10} {r['nll']:>8.4f} {r['ppl']:>8.2f} {r['delta_nll']:>+9.4f}  {ci:>20} {r['token_acc']:>8.4f}")

    w = {k: float(np.average([a[k] for a in attn_rows], weights=[a["n_items"] for a in attn_rows]))
         for k in ("prop", "nodes", "chars", "uniform_ref")}
    print("\nMasa de atención cross-attn (media capas/cabezas/targets no-pad):")
    print(f"  token de propiedad : {w['prop']:.5f}   (referencia uniforme 1/(1+N+L) = {w['uniform_ref']:.5f}"
          f"  ->  {w['prop'] / max(w['uniform_ref'], 1e-12):.2f}x)")
    print(f"  nodos de grafo     : {w['nodes']:.5f}")
    print(f"  caracteres SMILES  : {w['chars']:.5f}")

    out = Path(args.out) if args.out else REPO_ROOT / "results" / "mola" / f"ablate_prop_token_{dataset_name}_{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for k, val in w.items():
        df[f"attn_{k}"] = val
    df.insert(0, "dataset", dataset_name)
    df.insert(1, "split", args.split)
    df.to_csv(out, index=False)
    print(f"\nEscrito {out}")

    worst = max(abs(r["delta_nll"]) for r in rows if r["variant"] != "baseline")
    print(
        "\nLectura: si ningún ΔNLL cruza el 0 con holgura (IC95% conteniendo 0), el decoder "
        f"ignora el token. Mayor |ΔNLL| observado = {worst:.4f} nats/token."
    )


if __name__ == "__main__":
    main()
