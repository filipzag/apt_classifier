"""QLoRA fine-tuning of cribl-decision on the C1 technique question (``.venv-cribl``).

Training rows are ``cases_train.jsonl`` (public train splits, see ``export_cases.py``). Each
example is one C1 question with the case's candidates in a fresh random order (none_of_these last);
the loss is cross-entropy of the letter readout against the gold option(s), or none_of_these when
no gold is offered. The base stays frozen in 4-bit; only the LoRA trains. Checkpoints (adapter only)
go to ``data/models/cribl_ft/<arm>/step_<n>/`` every half epoch, each scored on the full dev split;
the log with every checkpoint's dev metrics goes to ``data/audits/cribl_ft/<arm>_train_log.json``.
The checkpoint is chosen on dev exact (ties: the earlier one) and never on CTI-HAL or test.
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
    read_jsonl,
    target_distribution,
)
from decision_model import Readout, load_model
from evaluate import metrics, run_split

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDITS = REPO_ROOT / "data" / "audits" / "cribl_ft"
MODELS = REPO_ROOT / "data" / "models" / "cribl_ft"


def schedule(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def example(row: dict[str, Any], rng: random.Random) -> tuple[list[dict[str, str]], list[float]]:
    candidates = list(row["candidates"])
    rng.shuffle(candidates)
    options = c1_options(candidates)
    return options, target_distribution(options, row["gold_ids"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--arm", required=True, help="run name, e.g. B1 or B2")
    parser.add_argument("--init", default="cribl", choices=["cribl", "fresh"])
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
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rows = [row for row in read_jsonl(args.cases_dir / "cases_train.jsonl") if row["candidates"]]
    rng.shuffle(rows)
    if args.limit_train:
        rows = rows[: args.limit_train]
    dev = read_jsonl(args.cases_dir / "cases_dev.jsonl")
    if args.dev_limit:
        dev = dev[: args.dev_limit]

    model, tokenizer, info = load_model(args.init, trainable=True)
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
        "examples_total": examples_total,
        "optimizer_steps": total_steps,
        "checkpoints": [],
        "skipped_long": 0,
    }
    log_path = AUDITS / f"{args.arm}_train_log.json"

    def save_log() -> None:
        log_path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")

    def checkpoint(seen: int) -> None:
        directory = models_dir / f"step_{seen}"
        model.save_pretrained(str(directory))
        model.eval()
        started = time.monotonic()
        dev_rows = run_split(readout, dev, AUDITS / f"{args.arm}_step{seen}_dev.jsonl", progress=0)
        report = metrics(dev, dev_rows)
        model.train()
        pooled = report["arm"]["pooled"]
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
                "dev_eval_s": round(time.monotonic() - started, 1),
            }
        )
        best = max(
            log["checkpoints"], key=lambda item: (item["dev"]["exact"], -item["examples_seen"])
        )
        log["selected"] = best["dir"]
        save_log()
        print(
            f"checkpoint {seen}: dev exact {pooled['exact']} parent-aware {pooled['parent_aware']} no-pick {pooled['no_pick']}",
            flush=True,
        )

    model.train()
    started = time.monotonic()
    seen = step = 0
    running: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    while seen < examples_total:
        order = list(rows)
        rng.shuffle(order)
        for row in order:
            if seen >= examples_total:
                break
            options, target = example(row, rng)
            try:
                input_ids = readout.encode(c1_state(row["text"]), C1_INSTRUCTIONS, options)
            except ValueError:
                input_ids = None
                log["skipped_long"] += 1
            if input_ids is not None:
                logits = readout.letter_logits(input_ids, len(options))
                wanted = torch.tensor(target, device=logits.device)
                loss = -(wanted * torch.log_softmax(logits, dim=-1)).sum()
                (loss / args.accum).backward()
                running.append(float(loss))
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
                        f"{elapsed / seen:.2f} s/example, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                        flush=True,
                    )
            if seen % every == 0 or seen == examples_total:
                log["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                log["train_s"] = round(time.monotonic() - started, 1)
                checkpoint(seen)
    log["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    save_log()
    print(f"selected on dev: {log['selected']}")


if __name__ == "__main__":
    main()
