"""QLoRA fine-tuning of cribl-decision on the C1 technique question (``.venv-cribl``).

Training rows are ``cases_train.jsonl`` (public train splits, see ``export_cases.py``). Each
example is one C1 question with the case's candidates in a fresh random order (none_of_these last);
the loss is cross-entropy of the letter readout against the gold option(s), or none_of_these when
no gold is offered. The base stays frozen in 4-bit; only the LoRA trains. Checkpoints (adapter only)
go to ``data/models/cribl_ft/<arm>/step_<n>/`` every half epoch, each scored on the full dev split;
the log with every checkpoint's dev metrics goes to ``data/audits/cribl_ft/<arm>_train_log.json``.
The checkpoint is chosen on dev exact (ties: the earlier one) and never on CTI-HAL or test.

Options for a continued arm (B3, prereg ``cribl_annoctr_2026-10-13.json``): ``--init`` a checkpoint
dir, ``--train-file`` another case file, ``--replay-file``/``--replay-count`` mixes a seeded sample
of a second file in, ``--family-target-datasets`` gives those datasets' rows
:func:`common.family_target_distribution` targets, ``--dev-file`` scores another dev split and
``--select`` picks on the never-abstain rule (:func:`common.forced_row`).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from common import (
    C1_INSTRUCTIONS,
    FORMATS,
    c1_options,
    c1_state,
    family_target_distribution,
    forced_row,
    read_jsonl,
    score_rows,
    target_distribution,
)
from decision_model import Readout, is_oom, load_model, wait_after_oom, with_oom_retry
from evaluate import metrics, run_split

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDITS = REPO_ROOT / "data" / "audits" / "cribl_ft"
MODELS = REPO_ROOT / "data" / "models" / "cribl_ft"
MAX_OOM_SKIPS = 50


def schedule(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


SELECT = {  # checkpoint selection on dev: (rule, metric)
    "exact": ("registered", "exact"),
    "forced_exact": ("never_abstain", "exact"),
    "forced_parent_aware": ("never_abstain", "parent_aware"),
}


def example(
    row: dict[str, Any], rng: random.Random, family: frozenset[str] = frozenset()
) -> tuple[list[dict[str, str]], list[float]]:
    candidates = list(row["candidates"])
    rng.shuffle(candidates)
    options = c1_options(candidates)
    target = family_target_distribution if row.get("dataset") in family else target_distribution
    return options, target(options, row["gold_ids"])


def resume_point(arm: str, directory: Path, log_path: Path) -> int:
    """The example count of checkpoint ``directory`` of ``arm``; refuses unless it is this
    arm's checkpoint, its log exists and nothing past it was written (dev predictions or
    later checkpoint dirs)."""
    directory = directory.resolve()
    if directory.parent != (MODELS / arm).resolve() or not directory.name.startswith("step_"):
        raise SystemExit(f"{directory} is not a checkpoint of arm {arm!r}")
    start = int(directory.name.removeprefix("step_"))
    if not log_path.exists() or not (directory / "adapter_model.safetensors").exists():
        raise SystemExit(f"cannot resume {arm!r}: missing log or adapter")
    later = [
        path
        for path in [*AUDITS.glob(f"{arm}_step*_dev.jsonl"), *(MODELS / arm).glob("step_*")]
        if int(path.name.split("step", 1)[1].lstrip("_").split("_")[0]) > start
    ]
    if later:
        raise SystemExit(f"outputs past step {start} exist: {[str(path) for path in later]}")
    return start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--arm", required=True, help="run name, e.g. B1 or B2")
    parser.add_argument("--init", default="cribl", help="cribl | fresh | <checkpoint dir>")
    parser.add_argument("--format", default="plain", choices=FORMATS)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accum", type=int, default=16, help="examples per optimizer step")
    parser.add_argument("--warmup", type=float, default=0.03, help="share of optimizer steps")
    parser.add_argument("--checkpoints-per-epoch", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit-train", type=int, help="first N shuffled train rows (pilot)")
    parser.add_argument("--dev-limit", type=int, help="score only the first N dev cases (pilot)")
    parser.add_argument("--cases-dir", type=Path, default=AUDITS)
    parser.add_argument("--train-file", default="cases_train.jsonl")
    parser.add_argument("--replay-file", help="mix a seeded sample of this case file in")
    parser.add_argument("--replay-count", type=int, default=0)
    parser.add_argument("--family-target-datasets", nargs="*", default=[])
    parser.add_argument("--dev-file", default="cases_dev.jsonl")
    parser.add_argument("--select", default="exact", choices=sorted(SELECT))
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="a checkpoint dir of this arm (<models>/<arm>/step_N): continue the same run from it",
    )
    args = parser.parse_args()
    if (
        args.init not in ("cribl", "fresh")
        and not (Path(args.init) / "adapter_config.json").exists()
    ):
        raise SystemExit(f"--init {args.init!r} is neither cribl, fresh nor a checkpoint dir")

    log_path = AUDITS / f"{args.arm}_train_log.json"
    start = 0
    if args.resume_from:
        start = resume_point(args.arm, args.resume_from, log_path)
    else:
        stale = [path for path in (log_path, MODELS / args.arm) if path.exists()] + sorted(
            AUDITS.glob(f"{args.arm}_step*_dev.jsonl")
        )
        if stale:  # checkpoint dev predictions resume by case id: never mix two runs
            raise SystemExit(
                f"arm {args.arm!r} already has outputs: {[str(path) for path in stale]}"
            )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rows = [row for row in read_jsonl(args.cases_dir / args.train_file) if row["candidates"]]
    replay: list[dict[str, Any]] = []
    if args.replay_file and args.replay_count:
        pool = [row for row in read_jsonl(args.cases_dir / args.replay_file) if row["candidates"]]
        replay = random.Random(f"{args.seed}:replay").sample(
            pool, min(args.replay_count, len(pool))
        )
        rows += replay
    family = frozenset(args.family_target_datasets)
    rng.shuffle(rows)
    if args.limit_train:
        rows = rows[: args.limit_train]
    dev = read_jsonl(args.cases_dir / args.dev_file)
    dev_order = {case["case_id"]: [item["id"] for item in case["candidates"][:10]] for case in dev}
    if args.dev_limit:
        dev = dev[: args.dev_limit]

    model, tokenizer, info = load_model(
        str(args.resume_from) if args.resume_from else args.init, trainable=True
    )
    readout = Readout(model, tokenizer, args.format, max_tokens=args.max_tokens)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    examples_total = int(len(rows) * args.epochs)
    total_steps = math.ceil(examples_total / args.accum)
    warmup = max(1, int(total_steps * args.warmup))
    every = max(1, len(rows) // args.checkpoints_per_epoch)
    models_dir = MODELS / args.arm
    models_dir.mkdir(parents=True, exist_ok=True)
    AUDITS.mkdir(parents=True, exist_ok=True)
    log: dict[str, Any] = {
        "arm": args.arm,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "args": {key: str(value) for key, value in vars(args).items()},
        "model": info,
        "train_rows": len(rows),
        "train_none_target": sum(1 for row in rows if not row["gold_in_candidates"]),
        "replay_rows": len(replay),
        "family_target_rows": sum(1 for row in rows if row.get("dataset") in family),
        "family_none_target": sum(
            1
            for row in rows
            if row.get("dataset") in family and not row.get("gold_family_in_candidates")
        ),
        "examples_total": examples_total,
        "optimizer_steps": total_steps,
        "checkpoints": [],
        "skipped_long": 0,
        "oom_skipped": 0,
    }
    if start:
        previous = json.loads(log_path.read_text(encoding="utf-8"))
        log = {
            **previous,
            "checkpoints": [c for c in previous["checkpoints"] if c["examples_seen"] <= start],
            "loss": [c for c in previous.get("loss", []) if c["examples_seen"] <= start],
            "resumed": {
                "from": str(args.resume_from),
                "examples_seen": start,
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "model": info,
                "replayed": "same seed and data/option order (the skipped examples consume the "
                "random generator as before); optimizer moments and dropout draws start afresh; "
                "the partial accumulation window at the checkpoint is lost",
            },
        }

    def save_log() -> None:
        log_path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")

    def checkpoint(seen: int) -> None:
        directory = models_dir / f"step_{seen}"
        model.save_pretrained(str(directory))
        model.eval()
        started = time.monotonic()
        dev_rows = with_oom_retry(
            lambda: run_split(
                readout, dev, AUDITS / f"{args.arm}_step{seen}_dev.jsonl", progress=0
            ),
            note="dev scoring",
        )
        report = metrics(dev, dev_rows)
        model.train()
        pooled = report["arm"]["pooled"]
        forced = score_rows([forced_row(row, dev_order[row["case_id"]]) for row in dev_rows])[
            "pooled"
        ]
        log["checkpoints"].append(
            {
                "examples_seen": seen,
                "epoch": round(seen / len(rows), 3),
                "dir": str(directory),
                "dev": {
                    "n": pooled["n"],
                    "exact": pooled["exact"],
                    "parent_aware": pooled["parent_aware"],
                    "no_pick": pooled["no_pick"],
                    "ece": pooled["ece"],
                },
                "dev_never_abstain": {key: forced[key] for key in ("exact", "parent_aware", "ece")},
                "dev_eval_s": round(time.monotonic() - started, 1),
            }
        )
        rule, metric = SELECT[args.select]
        block = "dev" if rule == "registered" else "dev_never_abstain"
        best = max(
            log["checkpoints"], key=lambda item: (item[block][metric], -item["examples_seen"])
        )
        log["selected"] = best["dir"]
        save_log()
        print(
            f"checkpoint {seen}: dev exact {pooled['exact']} parent-aware {pooled['parent_aware']} no-pick {pooled['no_pick']}"
            f" | never-abstain exact {forced['exact']} parent-aware {forced['parent_aware']}",
            flush=True,
        )

    model.train()
    started = time.monotonic()
    seen = 0
    step = start // args.accum
    running: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    while seen < examples_total:
        order = list(rows)
        rng.shuffle(order)
        for row in order:
            if seen >= examples_total:
                break
            options, target = example(row, rng, family)
            if seen < start:  # resumed: replay the order without training
                seen += 1
                continue
            try:
                input_ids = readout.encode(c1_state(row["text"]), C1_INSTRUCTIONS, options)
            except ValueError:
                input_ids = None
                log["skipped_long"] += 1
            oom = False
            if input_ids is not None:
                try:
                    logits = readout.letter_logits(input_ids, len(options))
                    wanted = torch.tensor(target, device=logits.device)
                    loss = -(wanted * torch.log_softmax(logits, dim=-1)).sum()
                    (loss / args.accum).backward()
                    running.append(float(loss))
                except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
                    if not is_oom(error) or log["oom_skipped"] >= MAX_OOM_SKIPS:
                        raise
                    oom = True
            if oom:  # another job took the GPU: drop this window's partial gradients, wait
                logits = loss = None
                log["oom_skipped"] += 1
                optimizer.zero_grad(set_to_none=True)
                wait_after_oom(f"training example {seen}")
            seen += 1
            if seen % args.accum == 0 or seen == examples_total:
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * schedule(step, total_steps, warmup)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    elapsed = time.monotonic() - started
                    mean = sum(running) / len(running)
                    log.setdefault("loss", []).append(
                        {
                            "step": step,
                            "examples_seen": seen,
                            "loss": round(mean, 5),
                            "elapsed_s": round(elapsed, 1),
                        }
                    )
                    running.clear()
                    print(
                        f"step {step}/{total_steps} loss {mean:.4f} "
                        f"{elapsed / (seen - start):.2f} s/example, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                        flush=True,
                    )
            if seen % every == 0 or seen == examples_total:
                log["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                log["train_s" if not start else "train_s_after_resume"] = round(
                    time.monotonic() - started, 1
                )
                checkpoint(seen)
    log["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    save_log()
    print(f"selected on dev: {log['selected']}")


if __name__ == "__main__":
    main()
