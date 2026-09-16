"""Does a prediction depend on which molecules share its batch?

    python scripts/batch_dependence.py --config mola
    python scripts/batch_dependence.py --config mola-fixed     # the control

Trains one model the way the benchmark trains it, then scores the SAME test molecules
several times, changing only how they are grouped: a few batch sizes, and several
shuffles at a fixed batch size. A model whose predictions depend only on their inputs
returns one number every time.

Why it matters. MoLA builds its SMILES branch with positional_smiles=False, which the
reference benchmark does not override. Under that setting sm_embed leaves the embedding
as [B, L, H] and enters a TransformerEncoderLayer with batch_first=False, which reads it
as [seq, batch, feature]: attention runs over the molecule axis, so each molecule attends
to the others in its batch at a fixed character position rather than over its own
characters. No labels cross -- this is not leakage of y -- but a test metric assumes
f(x_i) depends on x_i alone, and here it depends on the other 31 molecules that happened
to be grouped with it. The shuffle rows are the direct measurement of that.

--config mola-fixed runs the same architecture with the axis corrected, and is the
control: it must return the same number under every grouping. If it does not, the cause
is somewhere else and this script is measuring the wrong thing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from torch_geometric.loader import DataLoader as GeomDataLoader  # noqa: E402

from common.repro import TargetStandardizer, regression_metrics, seed_everything  # noqa: E402
from crossmodal_model.benchmark.pretrained_ablation import load_split_mola  # noqa: E402
from crossmodal_model.model.mola import MoLA  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="esol", choices=["esol", "freesolv", "lipo"])
    p.add_argument("--config", default="mola", choices=["mola", "mola-fixed"])
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-batches", type=int, nargs="+", default=[1, 4, 8, 32, 64])
    p.add_argument("--shuffles", type=int, default=5,
                   help="re-scorings at the training batch size with the test set in a "
                        "different order. Same molecules, same batch size: only the "
                        "grouping changes")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def score(model, dataset, batch_size, device, standardizer, shuffle_seed=None):
    """RMSE over the whole set, in the target's own units."""
    model.eval()
    order = list(range(len(dataset)))
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(order)
    data = [dataset[i] for i in order]
    loader = GeomDataLoader(data, batch_size=batch_size, shuffle=False)
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)[3].view(-1)
        preds.append(standardizer.inverse_transform(out).cpu())
        targets.append(batch.y.view(-1).cpu())
    return regression_metrics(torch.cat(preds), torch.cat(targets))["rmse"]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, deterministic=False)
    device = args.device

    splits, vocab = load_split_mola(args.dataset, args.seed)
    train_y = torch.tensor([float(d.y) for d in splits["train"]])
    standardizer = TargetStandardizer(enabled=True).fit(train_y)

    model = MoLA(
        graph_dim=splits["train"][0].x.size(1), sm_vocab_size=len(vocab),
        hidden_dim=args.hidden_dim, output_dim=1, num_layers=args.num_layers,
        positional_smiles=(args.config == "mola-fixed"), max_sm_len=100,
    ).to(device)
    print(f"{args.config} on {args.dataset} | positional_smiles="
          f"{args.config == 'mola-fixed'} | {sum(p.numel() for p in model.parameters()):,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.MSELoss()
    loader = GeomDataLoader(splits["train"], batch_size=args.batch_size, shuffle=True)
    for ep in range(args.epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch)[3].view(-1)
            loss = crit(out, standardizer.transform(batch.y.view(-1)))
            loss.backward()
            opt.step()
        if (ep + 1) % 20 == 0:
            print(f"  epoch {ep + 1}: train loss {loss.item():.4f}")

    test = splits["test"]
    print(f"\nscoring the same {len(test)} test molecules under different groupings\n")

    print(f"  {'batch size':>12} {'RMSE':>9}")
    by_size = {}
    for bs in args.eval_batches:
        by_size[bs] = score(model, test, bs, device, standardizer)
        print(f"  {bs:>12} {by_size[bs]:>9.4f}")

    print(f"\n  {'shuffle':>12} {'RMSE':>9}   (batch {args.batch_size}, same molecules)")
    shuffled = [score(model, test, args.batch_size, device, standardizer, shuffle_seed=s)
                for s in range(args.shuffles)]
    for s, r in enumerate(shuffled):
        print(f"  {s:>12} {r:>9.4f}")

    spread_size = max(by_size.values()) - min(by_size.values())
    spread_shuf = max(shuffled) - min(shuffled)
    print(f"\n  spread across batch sizes : {spread_size:.4f}")
    print(f"  spread across shuffles    : {spread_shuf:.4f}")
    # A tolerance, not zero: float reductions over different groupings differ in the last
    # bits, and that is not what this is looking for.
    tol = 1e-3
    if max(spread_size, spread_shuf) < tol:
        print(f"\n  Below {tol}: predictions do not depend on the grouping.")
    else:
        print(f"\n  The same model scores the same molecules differently depending on how")
        print(f"  they are batched. The reported RMSE is a property of the batching as")
        print(f"  much as of the model, and a single molecule cannot be scored on its own.")


if __name__ == "__main__":
    main()
