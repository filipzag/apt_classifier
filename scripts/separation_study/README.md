# What separates the ATT&CK groups: research study

Question: which kind of evidence puts the right group first, so that development effort goes
where the separation is? Development data only (ATT&CK itself, the vendor reports, the ATT&CK
campaign reports and the FBI/CISA advisories); CTI-HAL and Elastic are not read. Results and
reading: docs 21.3.24.

| File | What it does |
| --- | --- |
| `atlas.py` | ATT&CK alone: exclusive and rare techniques and software per group, nested profiles, ideal identifiability from the group's own techniques (exact ids against parent level), and ATT&CK's single-group campaigns ranked against the group profiles by technique list and by software (seconds). |
| `run_reports.sh` | Runs `scripts/attribution_benchmark.py` on the three development report sets at the `names`, `strict` and `tradecraft` anonymization levels with every member voting (the optional text, behaviour-text and group-network members at weight 1, so the learned combiner is inactive); each row records every member's rank of the actor (`classifier_truth_ranks`). Needs the GPU, ChromaDB and the technique head service; about 80 minutes; resumable; writes `data/audits/separation_study/` (gitignored). |
| `members.py` | Tables from those runs: per member and for three evidence-only rankings (named software by rarity, techniques by IDF cosine, rare techniques only) the actor's top-1/top-5/MRR, for all reports and split by whether the report names ATT&CK software (seconds). |
| `statements.py` | Statement level: every report statement against ATT&CK's procedure sentences (the behaviour-text index, each report's cited and copied texts hidden); per kind of statement (names software, command, file name or path, registry key, network indicator, victim sector or region, plain description) how often its nearest sentence is the actor's, against a permutation baseline (CPU, about 35 minutes for three levels). |

```bash
PYTHONPATH=. .venv/bin/python scripts/separation_study/atlas.py
scripts/separation_study/run_reports.sh
PYTHONPATH=. .venv/bin/python scripts/separation_study/members.py
PYTHONPATH=.:scripts .venv/bin/python scripts/separation_study/statements.py names strict tradecraft
```

Caveats: the vendor set and the campaigns already served member choices (docs 21.3.19-21.3.20),
so the text members' numbers on them lean optimistic; the statement tags are regular expressions
and overlap.
