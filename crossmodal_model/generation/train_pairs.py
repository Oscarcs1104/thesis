"""Train the lead-optimization generator: (M_a, delta property) -> M_b.

    encoder  HybridMoLA over M_a          graph + SMILES characters
    condition 4 delta-bin prefix tokens   logP / TPSA / QED / MW, dropped independently
    decoder  SELFIES of M_b               causal self-attention + cross-attention

The budget is set in STEPS, not epochs. The three ablation arms (graph+SMILES, graph
only, SMILES only) run sequentially on one GPU, and an arm that trained longer because
its epochs were cheaper would make the comparison about wall clock instead of about the
modalities. --max-steps is therefore the contract: every arm gets the same.

    python crossmodal_model/generation/train_pairs.py --max-steps 60000
    python crossmodal_model/generation/train_pairs.py --max-steps 60000 --no-use-smiles
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from torch_geometric.loader import DataLoader as GeomDataLoader  # noqa: E402

from common.repro import seed_everything  # noqa: E402
from common.wandb_utils import (  # noqa: E402
    add_wandb_args,
    wandb_finish,
    wandb_init,
    wandb_log,
    wandb_summary,
)
from crossmodal_model.generation.conditional_decoder import ConditionalMoleculeGenerator  # noqa: E402
from crossmodal_model.generation.pair_data import build_pair_datasets  # noqa: E402
from crossmodal_model.model.mola_hybrid import HybridMoLA  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-dir", type=str, default="data/moses")
    p.add_argument("--max-steps", type=int, default=60000,
                   help="the budget every ablation arm gets; this is what makes them comparable")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--ckpt-every-min", type=float, default=30.0)
    p.add_argument("--hidden-dim", type=int, default=256,
                   help="encoder+decoder width. Overridden by --init-encoder, which "
                        "must use whatever shape the checkpoint was trained in")
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--gin-hidden-mult", type=int, default=8,
                   help="widens the GIN update MLP's inner layer to hidden_dim * this. 8 matches "
                        "the SMILES branch's feedforward block, which defaults to "
                        "dim_feedforward=2048 against hidden 256. At 1 the graph branch "
                        "holds 450,816 parameters against the SMILES branch's 3,945,216, "
                        "so the modality ablation compares capacities, not modalities")
    p.add_argument("--decoder-layers", type=int, default=6)
    p.add_argument("--num-bins", type=int, default=20)
    p.add_argument("--cond-dropout", type=float, default=0.15)
    p.add_argument("--fusion-in-memory", action="store_true",
                   help="let MoLA's cross-layer fusion contribute 2L tokens to the "
                        "decoder's memory. Without it cross_attention and layer_weights "
                        "never run during generation -- 527,367 parameters allocated and "
                        "never trained -- and the modality ablation compares which raw "
                        "states enter the memory rather than anything about the fusion")
    p.add_argument("--max-sm-len", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--use-graph", dest="use_graph", action="store_true", default=True)
    p.add_argument("--no-use-graph", dest="use_graph", action="store_false")
    p.add_argument("--use-smiles", dest="use_smiles", action="store_true", default=True)
    p.add_argument("--no-use-smiles", dest="use_smiles", action="store_false")
    p.add_argument("--init-encoder", type=str, default=None,
                   help="HybridMoLA checkpoint from crossmodal_model/train/pretrain_moses.py "
                        "--arch hybrid. Only the encoder is taken; the regression head it "
                        "was pretrained with has no role here")
    p.add_argument("--freeze-encoder", action="store_true",
                   help="keep the pretrained encoder fixed. Weakens the claim from 'the fused "
                        "encoder makes the decoder work' to 'a frozen pretrained encoder helps'")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="checkpoints/pairs")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    add_wandb_args(p)
    p.set_defaults(wandb_project="thesis-generation-pairs")
    return p.parse_args()


def shift_targets(tgt: torch.Tensor, start_idx: int, end_idx: int, pad_idx: int):
    """[B, L] stored tokens -> (decoder input, decoder target), both [B, L+1].

    START/END are added here rather than stored, so 1.94M cached rows do not each carry
    two wasted slots and the special-token layout can change without re-encoding.
    """
    b, length = tgt.shape
    lengths = (tgt != pad_idx).sum(dim=1)
    dec_in = torch.full((b, length + 1), pad_idx, dtype=torch.long, device=tgt.device)
    dec_in[:, 0] = start_idx
    dec_in[:, 1:] = tgt
    dec_tgt = torch.full((b, length + 1), pad_idx, dtype=torch.long, device=tgt.device)
    dec_tgt[:, :length] = tgt
    dec_tgt[torch.arange(b, device=tgt.device), lengths] = end_idx
    return dec_in, dec_tgt


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, deterministic=False)
    device = args.device
    arm = ("graph+smiles" if args.use_graph and args.use_smiles
           else "graph-only" if args.use_graph else "smiles-only")
    # The init state is part of the identity, not a detail: the same arm trained from
    # scratch and from a pretrained encoder are two different models answering two
    # different questions, and naming them alike has the second silently overwrite the
    # first. The "+" is dropped for the same reason as in pretrain_moses.py -- a plus
    # sign in a path that travels through shell variables is an avoidable hazard.
    init_tag = "pretrained" if args.init_encoder else "scratch"
    fuse_tag = "_fused" if args.fusion_in_memory else ""
    run_name = args.run_name or f"pairs_{arm.replace('+', '_')}_{init_tag}{fuse_tag}_s{args.seed}"
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{run_name}.pt"
    if ckpt_path.exists():
        print(f"NOTE: {ckpt_path.name} already exists and will be overwritten. "
              f"Pass --run-name to keep both.")

    # Grouped by ablation arm so the three show up on one chart, which is the figure.
    run = wandb_init(args, config=vars(args), name=run_name, group="pairs-ablation",
                     tags=[arm, "pretrained-encoder" if args.init_encoder else "from-scratch"])

    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | VRAM {props.total_memory / 1e9:.0f} GB")

    corpus_dir = ROOT / args.corpus_dir
    train_ds, val_ds, _, cache, binners, vocab = build_pair_datasets(
        corpus_dir, num_bins=args.num_bins, max_sm_len=args.max_sm_len, seed=args.seed,
        workers=args.num_workers * 2, max_pairs=args.max_pairs, rebuild_cache=args.rebuild_cache,
    )
    pad_idx, start_idx, end_idx = vocab["pad_idx"], vocab["start_idx"], vocab["end_idx"]

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=device.startswith("cuda"),
                         persistent_workers=args.num_workers > 0)
    train_loader = GeomDataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = GeomDataLoader(val_ds, shuffle=False, **loader_kwargs)

    # Encoder dimensions come from the checkpoint, not from this script's defaults. The
    # pretrained weights only fit the shape they were trained in, and a mismatch here
    # fails at load time with a shape error rather than anywhere informative.
    hidden_dim, num_layers = args.hidden_dim, args.num_layers
    gin_mult = args.gin_hidden_mult
    # Por defecto fuera del bloque que carga el checkpoint: las filas desde cero no
    # tienen checkpoint del que leerlo y el modelo se construye igualmente.
    normalizar = False
    ck = None
    if args.init_encoder:
        ck = torch.load(args.init_encoder, map_location="cpu", weights_only=False)
        ck_args = ck.get("args", {})
        ck_hidden = int(ck_args.get("hidden_dim", hidden_dim))
        ck_layers = int(ck_args.get("num_layers", num_layers))
        if (ck_hidden, ck_layers) != (hidden_dim, num_layers):
            print(f"  encoder dims taken from the checkpoint: hidden {hidden_dim} -> {ck_hidden}, "
                  f"layers {num_layers} -> {ck_layers}")
        ck_mult = int(ck.get("gin_hidden_mult", ck_args.get("gin_hidden_mult", 1)))
        normalizar = bool(ck.get("normalize_branches",
                                 ck_args.get("normalize_branches", False)))
        if ck_mult != gin_mult:
            print(f"  GIN width multiplier taken from the checkpoint: {gin_mult} -> {ck_mult}")
        hidden_dim, num_layers, gin_mult = ck_hidden, ck_layers, ck_mult

    mola = HybridMoLA(
        sm_vocab_size=len(cache.char_vocab), hidden_dim=hidden_dim, output_dim=1,
        num_layers=num_layers, positional_smiles=True, max_sm_len=args.max_sm_len,
        use_graph=args.use_graph, use_smiles=args.use_smiles,
        gin_hidden_mult=gin_mult,
        normalize_branches=normalizar,
    )
    model = ConditionalMoleculeGenerator(
        mola, vocab_size=len(vocab["token_to_id"]), hidden_dim=hidden_dim, pad_idx=pad_idx,
        cond_vocab_sizes=[b.num_bins + 1 for b in binners.values()],
        cond_null_bins=[b.null_bin for b in binners.values()],
        cond_dropout=args.cond_dropout, fusion_in_memory=args.fusion_in_memory,
        decoder_layers=args.decoder_layers,
        max_len=args.max_sm_len + 32,
    ).to(device)
    if ck is not None:
        if ck.get("arch") != "hybrid":
            raise SystemExit(f"--init-encoder must be an --arch hybrid checkpoint, got "
                             f"{ck.get('arch')!r}. The pretrained-backbone architectures use a "
                             f"different featurization and character vocabulary.")
        if ck.get("char_vocab") != cache.char_vocab:
            raise SystemExit("the checkpoint's character vocabulary differs from this corpus's. "
                             "The SMILES embedding rows would mean different characters.")
        if (ck.get("use_graph"), ck.get("use_smiles")) != (args.use_graph, args.use_smiles):
            raise SystemExit(f"checkpoint arm is graph={ck.get('use_graph')} "
                             f"smiles={ck.get('use_smiles')}, this run is graph={args.use_graph} "
                             f"smiles={args.use_smiles}. Each arm must load its own.")
        enc = {k[len("encoder."):]: v for k, v in ck["model_state_dict"].items()
               if k.startswith("encoder.")}
        missing, unexpected = model.mola.encoder.load_state_dict(enc, strict=False)
        if missing or unexpected:
            raise SystemExit("encoder init did not load cleanly\n"
                             f"  missing:    {list(missing)}\n"
                             f"  unexpected: {list(unexpected)}")
        print(f"  encoder initialized from {args.init_encoder} (step {ck.get('step')})")
        if args.freeze_encoder:
            for prm in model.mola.encoder.parameters():
                prm.requires_grad = False
            print("  encoder frozen")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\narm={arm}  params={n_params:,}  budget={args.max_steps:,} steps "
          f"x batch {args.batch_size} = {args.max_steps * args.batch_size:,} examples")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay,
                                  fused=device.startswith("cuda"))

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        return args.lr * max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item()))

    use_amp = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    if use_amp:
        print("  bf16 autocast enabled")

    def batch_loss(batch) -> torch.Tensor:
        dec_in, dec_tgt = shift_targets(batch.tgt, start_idx, end_idx, pad_idx)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(batch, dec_in, batch.cond)
            return criterion(logits.reshape(-1, logits.size(-1)).float(), dec_tgt.reshape(-1))

    @torch.no_grad()
    def evaluate() -> Dict[str, float]:
        model.eval()
        total, n, correct, tokens = 0.0, 0, 0, 0
        for i, batch in enumerate(val_loader):
            if i >= args.eval_batches:
                break
            batch = batch.to(device, non_blocking=True)
            dec_in, dec_tgt = shift_targets(batch.tgt, start_idx, end_idx, pad_idx)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(batch, dec_in, batch.cond)
            loss = criterion(logits.reshape(-1, logits.size(-1)).float(), dec_tgt.reshape(-1))
            keep = dec_tgt != pad_idx
            correct += int(((logits.argmax(-1) == dec_tgt) & keep).sum())
            tokens += int(keep.sum())
            total += loss.item()
            n += 1
        model.train()
        return {"loss": total / max(n, 1), "token_acc": correct / max(tokens, 1)}

    history, step, best_val = [], 0, float("inf")
    start = time.time()
    last_ckpt = start
    model.train()
    print()
    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break
            batch = batch.to(device, non_blocking=True)
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            loss = batch_loss(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step += 1

            if step % args.eval_every == 0 or step == args.max_steps:
                val = evaluate()
                elapsed = time.time() - start
                rate = step / max(elapsed, 1e-9)
                eta = (args.max_steps - step) / max(rate, 1e-9)
                print(f"step {step:>7,}/{args.max_steps:,} | train {loss.item():.4f} | "
                      f"val {val['loss']:.4f} acc {val['token_acc']:.3f} | "
                      f"{rate:.1f} it/s | eta {eta / 60:.0f} min", flush=True)
                history.append({"step": step, "train_loss": loss.item(), **val})
                best_val = min(best_val, val["loss"])
                wandb_log(run, {"train/loss": loss.item(), "val/loss": val["loss"],
                                "val/token_acc": val["token_acc"],
                                "lr": optimizer.param_groups[0]["lr"]}, step=step)

            # Checkpoint on a wall-clock timer, not a step count: a job lost at hour 4
            # to a node failure should cost thirty minutes, not the whole arm.
            if time.time() - last_ckpt > args.ckpt_every_min * 60 or step == args.max_steps:
                torch.save({
                    "model_state_dict": model.state_dict(), "step": step, "args": vars(args),
                    # Explicit, because --init-encoder overrides these and vars(args)
                    # keeps the command line's value: anything rebuilding the model from
                    # args alone would get the wrong width and fail at load.
                    "hidden_dim": hidden_dim, "num_layers": num_layers,
                    "gin_hidden_mult": gin_mult,
                    "normalize_branches": normalizar,
                    "fusion_in_memory": bool(args.fusion_in_memory),
                    "arm": arm, "vocab": vocab, "char_vocab": cache.char_vocab,
                    "binners": {k: v.state_dict() for k, v in binners.items()},
                    "history": history,
                }, ckpt_path)
                last_ckpt = time.time()

    elapsed = time.time() - start
    print(f"\nDone: {step:,} steps in {elapsed / 60:.0f} min | best val loss {best_val:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    wandb_summary(run, {"best_val_loss": best_val, "arm": arm, "steps": step,
                        "params": n_params, "checkpoint": str(ckpt_path)})
    wandb_finish(run)
    (out_dir / f"{run_name}_history.json").write_text(
        json.dumps({"arm": arm, "params": n_params, "elapsed_s": elapsed, "history": history}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
