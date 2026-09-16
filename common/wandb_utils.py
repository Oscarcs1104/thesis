"""Thin optional Weights & Biases helper shared by the training entrypoints.

Only touched when --use-wandb is passed, so wandb stays an optional dependency.

Runs launched inside one process (the benchmark starts one per grid cell) need
reinit=True and a shared `group`, otherwise wandb either refuses the second init or
scatters nine cells across nine unrelated runs.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence


def add_wandb_args(parser) -> None:
    parser.add_argument("--use-wandb", action="store_true", help="Log this run to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="thesis-multimodal")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="group related runs (e.g. every cell of one benchmark grid)")


def wandb_init(
    args,
    config: dict,
    name: Optional[str] = None,
    group: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
) -> Optional[Any]:
    if not getattr(args, "use_wandb", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("--use-wandb requires the wandb package: pip install wandb") from exc

    # Compute nodes often have no outbound network. Offline runs still record everything
    # and are pushed later with `wandb sync`; failing the job for that would be absurd.
    if os.environ.get("WANDB_MODE") == "offline":
        print("  wandb: offline mode, sync later with `wandb sync <dir>`")

    return wandb.init(
        project=args.wandb_project,
        name=name or getattr(args, "wandb_run_name", None),
        entity=getattr(args, "wandb_entity", None),
        group=group or getattr(args, "wandb_group", None),
        tags=list(tags) if tags else None,
        config=config,
        reinit=True,
    )


def wandb_log(run: Optional[Any], metrics: dict, step: Optional[int] = None) -> None:
    if run is not None:
        run.log(metrics, step=step)


def wandb_summary(run: Optional[Any], values: dict) -> None:
    """Final numbers, so a grid can be read off the runs table without opening charts."""
    if run is not None:
        for key, value in values.items():
            run.summary[key] = value


def wandb_finish(run: Optional[Any]) -> None:
    if run is not None:
        run.finish()
