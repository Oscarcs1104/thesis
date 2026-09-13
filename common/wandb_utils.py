"""Thin optional Weights & Biases helper shared by the training/pretraining entrypoints.

Only touched when --use-wandb is passed, so wandb stays an optional dependency.
"""

from __future__ import annotations

from typing import Any, Optional


def add_wandb_args(parser) -> None:
    parser.add_argument("--use-wandb", action="store_true", help="Log this run to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="thesis-multimodal")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)


def wandb_init(args, config: dict) -> Optional[Any]:
    if not getattr(args, "use_wandb", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("--use-wandb requires the wandb package: pip install wandb") from exc
    return wandb.init(project=args.wandb_project, name=args.wandb_run_name, entity=args.wandb_entity, config=config)


def wandb_log(run: Optional[Any], metrics: dict, step: Optional[int] = None) -> None:
    if run is not None:
        run.log(metrics, step=step)


def wandb_finish(run: Optional[Any]) -> None:
    if run is not None:
        run.finish()
