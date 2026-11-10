# Dated knowledge store: design

Status: design draft, 2026-11-03. Nothing is built yet. Code references point at branch
`claude/finetune-cribl-mitre-attack-8aa2d5`, which holds the current product (120 commits past
`frontend-next`); the build should start from there. Background: the research note
[ATTRIBUTION_STATE_OF_THE_ART.md](ATTRIBUTION_STATE_OF_THE_ART.md) (section 4.1) and the technical
documentation, sections 21.3.28 to 21.3.31.

## The idea

Vestige reads the public research reports an organisation chooses to follow. It keeps the facts in
them as dated records, each with a link, for example "ESET, 12 May 2026: HOOKEDGE is used by
APT28". When it analyses a case, it uses only what was known before that case's date. ATT&CK stays
the backbone. The store adds what ATT&CK has not caught up with:

- new malware tied to an actor;
- new names for actors;
- how each actor has operated in recent reports.

Everything stays on the machine and runs without an OpenRouter key.

## Why

- **Recent knowledge is where the product loses.** On the fresh reports the product put the right
  group first on 19 of 44 reports and GPT-5.6 Sol on 24 (21.3.28). On the 11 reports the model got
  and the product missed, the right group was never in the product's top five. The model's reasons
  cite knowledge ATT&CK does not hold. About 8 of them rest on malware tied to the actor in recent
  reports (HOOKEDGE, the Ptero tools, NotDoor, TOSHIS); about 3 rest on vendor names (21.3.29).
- **The open lists don't hold that knowledge.**
  - MISP adds 58 malware-to-group links to ATT&CK's 1,166 and covers 1 of the 11 misses.
  - Malpedia covers at most 3, and its licence is non-commercial (21.3.29).
- **ATT&CK updates too slowly to fill the gap.** It updates twice a year and adds about 60 to 170
  group-software links a year. In version 20.0, 88 of its links date from 2025 and 59 from 2026.
- **Vendors publish far more than ATT&CK takes in.** The fresh-set builder's sources published
  about 160 posts a month in the first listing.
- **Habits matter as well as malware.** With malware names hidden, the model still beat the product:
  23 of 44 against 10 (21.3.30). It knows each actor's habits from years of reports. That is the
  second thing the store should hold: recent procedure text for each actor.

## What the store holds

| Record | Example | Comes from | Used for |
|---|---|---|---|
| Source | URL, publisher, title, publication date, last update, content hash, licence class | every fetched post | dating and linking every fact |
| Actor and its names | APT28 = Fancy Bear = BlueDelta = Forest Blizzard; Storm-1175 | ATT&CK, the MISP threat-actor list, Microsoft's name mapping, "tracked as" sentences | name matching, masking, labels |
| Software and its names | HOOKEDGE (backdoor), first seen 2026-05-12 | ATT&CK, code names introduced in reports | finding new malware names in a case |
| Uses link | APT28 uses HOOKEDGE: ESET 2026-05-12, Sekoia 2026-06-01 | sentences and single-actor reports | the new software member |
| Overlap or same-as claim | "overlaps with", "also tracked as" | sentences | shown to analysts; merges only under the rule below |
| Procedure text | the anonymised action sentences of single-actor reports, credited to that actor with the date | single-actor reports | the new text member |
| Later: artefacts and indicators | mutex, named pipe, PDB path, registry key, user agent; hashes and domains with dates | indicator sections and sentences | exact-match evidence (a later step) |

Every fact keeps:

- its source;
- the position of the sentence it came from (the text itself stays local);
- how it was read (a rule or the local model);
- the source's own hedging: confirmed, likely or possible.

## Where the reports come from

- **Sources.** The 33 sources already configured in `scripts/prepare_fresh_reports.py`
  (`SOURCES`). They are the sitemaps and feeds of 32 vendor research blogs plus CISA advisories,
  all approved by the product owner for the fresh sets. A new source needs the owner's approval.
  Analysts cannot add URLs; this is the rule of the Live Intel layer (docs 24.1).
- **Manners.**
  - Plain requests only, one request per second per site.
  - robots.txt is honoured. The builder does not check it today, so this is new code.
  - Refreshes use conditional requests (ETag and Last-Modified).
  - A site that refuses is dropped and logged, never worked around (21.3.31).
- **Kept out.**
  - Elastic Security Labs stays out while the Elastic benchmark is in use.
  - Posts in the Elastic sets are excluded by URL. CTI-HAL rows carry no URL or date (found
    while building step 1), so a CTI-HAL report's own post is found by statement overlap when
    that benchmark reads the store, and hidden then.
- **Identity.** The store names itself in its requests ("VestigeKnowledgeStore/1.0 …"), so
  robots.txt rules can address it; the fresh-set builder sent a browser string. A site that
  refuses the honest name is dropped, not retried as a browser.
- **Backfill: posts from 2022-01-01 onwards.** Sitemaps list a site's full history.
  - Expect roughly 10,000 to 12,000 posts, a few hours of downloading at the polite rate, and about
    3 to 4 GB of HTML. For comparison, the two fresh sets were 2,900 pages and 890 MB.
  - The HTML is kept gzipped (roughly a quarter of its size), so pages can be read again when
    the parser improves; the parsed lines sit compressed in the database.
- **First backfill (2026-11-03, step 1 done):**
  - 12,776 posts listed and 11,103 stored. Of these, 10,725 are dated and usable: 1,276 from
    2022, 1,391 from 2023, 1,319 from 2024, 1,698 from 2025, 1,834 from 2026, and the rest
    older. The store takes 0.44 GB of gzipped pages and a 71 MB database.
  - Out, never worked around:
    - AhnLab ASEC refuses its robots.txt to the reader.
    - Silent Push refuses its sitemaps.
    - Fortinet's robots.txt forbids its blog.

    The 82 Silent Push and Fortinet posts the fresh-set builder had downloaded (browser name, no
    robots.txt check) are kept on disk but left out of every view.
  - Limited: ESET's feed gives only its latest 100 posts, and it has no sitemap. Bitdefender
    gives 15 posts.
  - Rate-limited at one request per second: Check Point, SentinelOne and Zscaler (1,669 posts
    waiting). They are read again after a day, at 10 seconds between requests.
  - 214 posts have no usable date (mostly Rapid7, whose site migration re-dated its old posts)
    and appear in no dated view. 741 posts were modified more than 30 days after publication and
    count from their modification date. ATT&CK 20.0 cites 322 of the posts.
- **Daily refresh** after the backfill: tens of posts a day, a few minutes.
- **Name lists.**

  | List | Licence | Use |
  |---|---|---|
  | ATT&CK | — | as today |
  | MISP galaxy threat-actor list | CC0 / BSD-2 | can ship; already at `data/raw/misp_galaxy_24877de2/` |
  | [Microsoft's actor-name mapping](https://github.com/microsoft/mstic/tree/master/PublicFeeds/ThreatActorNaming) (181 actors, updated 2026-10-22) | CC BY 4.0 | can ship with attribution |
  | Malpedia | CC BY-NC-SA | evaluation only, never in the product |
  | ORKL (a public report library) | terms not found | not used until checked |

- **Licence of the text.** The report text belongs to each vendor; CISA advisories are US
  government works. The store database, its text and every model file built from it stay local
  and gitignored, as the fresh sets do. The product ships the code, the source list and the name
  lists. Each installation builds its own store.

## How a report is read

Everything below runs locally and spends no OpenRouter credit.

1. **Parse and date.** `prepare_vendor_reports.parse_page` gives the statements and
   `prepare_fresh_reports.page_date` the publication date. Some posts are updated more than 30
   days after publication (sitemap lastmod, or "(Updated …)" in the title). They are dated by the
   update, because their text may hold later knowledge.
2. **Duplicates and retellings.** These are the same post at two addresses, translations and
   syndicated copies, found with the builder's rule (statement overlap of at least 0.5). Each keeps
   one record, dated to the earliest copy. A post that links to another store post for a claim is
   credited to the original publisher.
3. **What the report is about.** The registered label rule (`label_report`) marks a report as
   single-actor when one actor:
   - is named at least three times;
   - is named twice as often as any other actor;
   - appears in the title or the first ten statements.

   Every other report is multi-actor or has no actor. The label reviews found the rule's errors
   are mostly roundups and ambiguous MISP synonyms ("Alibaba"). The store drops ambiguous names
   the same way.
4. **Names in the text.**
   - Actor names come from the dictionary: ATT&CK, MISP, Microsoft, and names the store has learned.
   - Software names come from ATT&CK, plus new code names found by the behaviour masker's rules
     (`report_anonymizer._BehaviorMask`): code-name shapes such as HOOKEDGE or PteroSand, and names
     the report gives a malware noun ("a backdoor we named X").
   - Ordinary English words never count (`ordinary_words`). The software matcher's inner-capitals
     rule applies.
5. **Uses links, rules first.**
   - A sentence that names an actor and a software with a use verb (deployed, used, delivered,
     dropped, installed, relied on) gives a uses link.
   - A single-actor report gives a report-level uses link for a malware it names at least twice,
     provided the malware is never next to another actor's name or a comparison phrase ("similar
     to", "previously attributed to", "overlaps with", "unlike").
   - Comparison sentences, multi-actor reports and cases where the rules disagree go to the local
     model (Qwen3.5 4B on the GPU). It uses the technique verifier's reason-first prompt style and
     token-probability cut-off (16.4a). Below the cut-off the link is not stored. That cut-off has
     proved fragile, so the quality sample below checks it.
   - Hedging words map to confirmed, likely or possible. "Possible" links never count towards
     naming.
   - *As built (2026-11-03, technical documentation 12.7):*
     - The report rule also needs one sentence that says the software is used.
     - Lures, leaks and publications ("released by") are unclear, like comparisons.
     - Names a post introduces count only when a quarter of the earlier posts that mention them
       introduce them as malware.
     - The model step (technical documentation 12.7) runs on the GPU only. It has answered all
       2,379 unclear links and confirmed 980. The laptop answered some; a rented GPU answered
       the rest after the laptop throttled.
     - 92 of 100 hand-checked confirmations are right.
     - The swapped-names test shows that the answers follow the text. Report-link questions,
       however, take the report's subject as given.
6. **Same-as and overlap claims.** Sentences such as "also tracked as", "overlaps with" and
   "linked to" are stored as claims. Two actors are merged only when ATT&CK, MISP or Microsoft says
   they are the same, or when two publishers say "same as". An overlap never merges them; this is
   the documented-overlap rule of the naming gate (10.11).
7. **Procedure text.** This is the statements of single-actor reports, prepared as follows:
   - the strict anonymiser is applied (actor names and cluster codes become "the adversary");
   - indicator lists and boilerplate are dropped. Boilerplate means sentences a publisher repeats
     across posts about different actors, such as "contact our team" or product names;
   - at most 60 statements are kept per report, so one long report does not dominate.

**Corroboration.**

- A uses link counts towards naming only when at least two independent publishers state it. The
  same vendor twice counts once.
- One publisher is enough to move a group up the shortlist, but not to name it.
- When publishers disagree (ESET says A, Talos says B), the software counts as shared by both
  actors.

**Quality is checked before anything is used.**

- Check by hand a random sample of 200 stored links, spread across the ways they were read.
  Optionally, GPT-5.6 gives a second opinion for about $1.
- Target: at least 90% of the links that count towards naming are right.
- *Result (2026-11-03, rules only, two samples of 200 outside the decisive window):*
  - 85% of all active links are right.
  - Links that can name, meaning pairs confirmed by two publishers: 109 of 117 right (93%).
- Side benefit: the same use-or-mention rules can be tried on case text. They address the open
  problem of a software mention attributed to another actor being read as use (noted after
  21.3.27).

## How the product uses it

- **A dated view.** `StoreView(as_of)` returns only facts from sources published before a given
  date. It returns them in the same shapes as the knowledge base (`software_groups`,
  `group_software`, aliases). In the app the date is today, or the case date if one is set. In
  every benchmark it is the report's own date.
- **Two new ensemble members**, both off by default. They follow the free-text member's pattern:
  at weight 0 they are not loaded, so the current combiner and its fingerprint stay valid.
  - *Store software* applies the software member's rarity rule to the store's uses links that
    ATT&CK lacks. It also matches new code names in the case text.
    - The naming gate counts it in the same family as the ATT&CK software member, so the two
      cannot pass as two independent kinds of evidence.
    - Only corroborated links count for naming.
  - *Store text* is the free-text member's TF-IDF over per-actor documents built from the store's
    procedure text: `TextGroupModel` with store records added, and `without` for slicing by date.
    Like the other text members it never counts as independent evidence
    (`NOT_INDEPENDENT_EVIDENCE`).

  At weight above 0, each member needs its own combiner (columns through
  `combiner.OPTIONAL_CLASSIFIERS`), probability floor and confidence labels, all fitted on the
  development lists.
- **Candidates stay the ATT&CK groups in the first version.** About half of the fresh reports (54
  of 99) were about actors ATT&CK does not document. For these the analyst sees a line such as
  "the store knows a matching actor outside ATT&CK: Storm-1175 (Microsoft, 3 reports)", without a
  name in the verdict. Naming such actors is a later step with its own test.
- **What the analyst sees.**
  - In the candidate dossier: every store fact the case touched, with publisher, date and link.
  - A status line for the store: last refresh, failed sources, number of facts.
- **Later, with a cloud key.** The store gives the "LLM proposes, Vestige checks" path (21.3.31)
  a dated dossier for the shortlist. The model must cite store facts, and Vestige checks that the
  cited facts exist and predate the case. This also fixes the problem of the model inventing
  ATT&CK ids (21.3.28).

## Keeping the test honest

| Leak | How it would flatter a result | Fix |
|---|---|---|
| The test report itself, or a copy of it, is in the store | a trivial match | facts only from posts published before the report's date; copies found by statement overlap; updated posts dated by their last update |
| Name lists from the future | a 2026 name links a 2025 cluster | the name lists pinned before the test window. **Registered 2026-11-04:** ATT&CK 16.1's names and aliases (the last release before 2025; each group's own ATT&CK 20.0 name also counts) and the MISP galaxy at commit 3735fd37 (2024-12-24). It has 793 actors, against 1,113 in October 2026. Microsoft's list is not used |
| Names the masking misses | the store matches the actor's own name | test texts masked with the full dictionary, including every name the store learned |
| ATT&CK fold cases are written from the same vendor reports | fold gains are fake | folds used only for the removed-actor safety check; the removed actor is deleted from the store too, and store posts that ATT&CK cites for a held-out sentence are hidden |
| ATT&CK 20.0 knows facts published after the test reports | flatters the baseline, so understates the store's gain | 147 of ATT&CK's 1,166 group-software links were created in 2025–2026; for each test report, count the right group's links ATT&CK added after the report's date |
| Publisher style | an actor covered mostly by one vendor is matched by that vendor's wording | remove boilerplate; on the development set, check that the store text member does not prefer actors covered by the test report's own publisher |
| The local model's own memory | the 4B model adds links it remembers | the relation prompt accepts only what the sentence says; re-run a sample with actor names swapped to check the answers follow the text. **Done 2026-11-03:** they follow the text (93 of 100 kept), but report links trust the subject the question states (12.7) |

## How we test it

- **Development.** All choices are made on these sets, with the store sliced to each report's
  date:
  - the first fresh set (44 known reports, 54 open-set);
  - the second fresh set (12 known, 9 open-set);
  - the vendor and campaign sets used for the procedure member.

  CTI-HAL and Elastic are used only to check that no new wrong name appears, never for tuning.
- **Early stop, after the backfill and before any member is built.** Take the 11 fresh reports the
  model got and the product missed. Does the store, as of each report's date, hold the link the
  model used? If it holds fewer than 3 of the 8 malware links, stop and report back.
- **Decisive test, registered before anything is built on it.**
  - Reports from **2025-01-01 to 2026-02-28** from the same sources, built with the same builder
    and the same label review.
  - None may be cited by ATT&CK 20.0 or be in any other set.
  - Nobody has looked at that period: the fresh listings began on 2026-02-20. *Checked at
    registration (2026-11-04):* a scan of every session transcript and store result file
    found 47 of the window's posts shown before, mostly while the fresh sets were built and
    during the research. Five more support links the early-stop output showed. All 52 are left
    out.
  - Expect roughly 70 to 100 known reports and a similar number of open-set reports (the fresh
    rate was about 6 known reports a month after review). A listing run takes minutes and gives
    the real number before registration. *Frozen 2026-11-04:* 48 known reports on 25 groups and
    69 open-set reports after the label review (docs 12.8), so every reading is descriptive
    only.
  - Posts from November 2026 onwards are collected as a confirmation set. The "LLM proposes" test
    can share them.
- **The two systems compared.** The default product, and the product with the store (registered
  weights, combiner and floor). Both read the same statements.
- **Readings.**
  - **Primary: behaviour** (the product owner's choice, 2026-11-04), with malware, tool and
    operation names hidden too. Only the store's procedure text can help there.
  - Secondary: strict, with actor names hidden and malware names kept. This is the case an analyst
    usually has.
- **Measures.**
  - Right group first on known reports, compared report by report, with a bootstrap interval and
    an exact McNemar test.
  - Right group in the top five.
  - Names given, and how many are right and wrong, on known and open-set reports.
- **Adoption.** Both conditions must hold:
  - first choice improves, with the interval above zero;
  - no new wrong name appears on the new set, on the ATT&CK folds (fresh seeds, removed actors),
    on CTI-HAL or on Elastic.

  With fewer than 60 known reports, every reading is descriptive only.

## Step 5b (registered 2026-11-09): the combiner learns from dated store posts

Registration: `docs/preregistration/store_training_lists_2026-11-09.json` (registered
2026-11-09). Built and run 2026-11-09 to 2026-11-10; the result and its review are at the end of
this section.

**Why.** Store text alone is the strongest member on the fresh reports at the behaviour reading
(20/44 and 7/12). The combiner still barely uses it, because it learns mostly from ATT&CK folds,
where store text is weak (28–31 of 155 first). Real reports read the way the product reads them
should teach it how far to trust store text.

**Training posts.**
- Store posts published before 2025. Nothing from the decisive window (2025-01 to 2026-02) is read,
  either as a case or as store content.
- Built with the fresh-set builder from the store: same label rule, masking, and cap of 8 posts per
  group. The decisive set's label review runs before any system reads them.
- Posts ATT&CK cites are left out, and so are the URLs and retellings of every other set (CTI-HAL
  and Elastic included).
- Open-set posts become real "none of these" lists for the combiner's null option.
- Pool before the cap, exclusions and review: 294 known posts on 79 groups (before 2022: 83; 2022:
  74; 2023: 61; 2024: 76) and 167 open-set posts. Expected: about 180–230 known.
- Built and reviewed 2026-11-09 (`data/benchmarks/store_train/`, hashes in the registration's
  build record): 95 posts excluded by the label review, never relabelled; 194 known posts on 72
  groups and 132 open-set posts on 97 actors. The 2024 choice slice holds 61 known and 47 open-set
  posts.

**How each post is read.**
- The store: the pre-2025 pinned copy, only posts from before the post's own date, without the post
  and its copies.
- ATT&CK members: ATT&CK 20.0, with the procedure examples that copy the post hidden. Older ATT&CK
  releases would mean rebuilding every index and model per release.
- Two readings: strict and behaviour.

**Choosing the weight.**
- Store lists get their own weight (0, 3, 10, 30). The existing real-report lists keep 10.
- A time split decides: train on posts before 2024, score on the 2024 posts. Most behaviour-reading
  first places wins.
- A guard stops any choice that drops ATT&CK fold top-1 by more than 0.02.
- If weight 0 wins, the step stops.

**Naming safety.**
- Floor and labels are re-chosen on the design seeds.
- The go rule: more development reports first at the behaviour reading (fresh, fresh2, vendor) and
  no wrong name the current product does not give. This covers those sets at both readings and the
  design seeds.
- The last wrong name came at combiner probability 0.92, so only that check, not a floor, guards
  against it.

**Deciding test.** The registered confirmation set (posts from 2026-11-03, scored at 30 known,
descriptive) needs the refresh. It did not run from 2026-11-04 to 2026-11-09; the run on 2026-11-09
added 27 posts and lost nothing. It is run by hand (`make store-update`), not scheduled; a run at
least every 60 days loses nothing. Sites that refuse or rate-limit are left alone (2026-11-09: Check
Point, SentinelOne, Zscaler, Silent Push, AhnLab ASEC); Trend Micro listed nothing after 2026-09-25.
At about 4–7 usable known posts a month:
- 30 by about March–June 2027;
- 60, the adoption minimum, by about mid-2027 to early 2028;
- 100 by about 2028.

**Cost.**
- No OpenRouter credit.
- About 10–11 hours of laptop GPU: lists about 7–8 h, design seeds about 50 min, development check
  about 1.6 h. A rented GPU would cost roughly $4–8.
- About 300–350 label reviews.

**Result (2026-11-10): store weight 0, so no combiner and no decisive run.**
- Lists: 1,025 from the store posts (191 known and 131 open-set posts at both readings, 381
  removed-actor lists). Run on the laptop GPU from 00:40 to 06:25, no credit.
- Time split: the 60 known posts of 2024, scored by a combiner fitted without them.
  - Every weight put 14–15 of the 60 first at the behaviour reading (weight 0: 15) and 28–30 at
    strict.
  - Weights 10 and 30 moved the right group up a little (mean reciprocal rank 0.34 → 0.35–0.37)
    but cost ATT&CK fold top-1 0.90 → 0.85–0.87, outside the guard.
  - Weight 3 passed the guard and put no more reports first.
- The rule picked weight 0, so no combiner was written. The design seeds, floor, labels and
  development check were not run. Defaults are unchanged.
- For scale: the right group was in the shortlist for 37 of the 60 posts at the behaviour reading
  (45 at strict).
- Report: `data/knowledge_store/step5b/attribution_combiner_store_lists_cv.json`.

**Review (2026-11-10, development data, made after the result).**
- The stop is the rule applied correctly. But 60 posts can only show a jump of about 8 first
  places, so the test had little power.
- Old posts are not like new ones. A post from 2019 meets a thin store, while ATT&CK 20.0 already
  documents its activity. The right group had no store text at all for 20 of the 34 posts up to
  2019, against 5 of the 60 posts of 2024. So the lists teach the combiner to trust the ATT&CK
  members.
- 643 of the 916 usable lists say "none of these". They teach caution only, and a higher naming
  floor on today's combiner does that better. At the same 2,151 right names on held-out fold cases
  it gives 196 wrong names (floor 0.44), against 293 with the store lists.
- The combiner cannot reach what store text knows:
  - it only reorders the fused top 20, and in 4 of store text's 10 own wins the right group was
    outside it;
  - a good store-text answer shows in its lead over its own second choice, and the combiner has no
    such feature.
- Three routes through the fusion or a combiner have now failed for the same reasons, so no fourth
  is proposed. What did pay on development data is a ranking rule outside the combiner (next
  section).

## Build order

| Step | What | Compute | Result |
|---|---|---|---|
| 1 | Store and collector: move the builder's listing, download, parse and dating code into `apt_classifier/knowledge_store/`; SQLite store; robots.txt; backfill from 2022 | download a few hours in the background; parsing about 1 hour of CPU | the store and a status report |
| 2 | Names and links: dictionaries, code names, rule-based links, local model for unclear cases, the 200-link quality sample | roughly 5–10 GPU hours in the background, with a watchdog for the nightly driver update | links with sources; share of checked links that are right. **Rules done 2026-11-03** (3,945 active links, 2,379 unclear; 93% of naming links right); **model step done 2026-11-03** (980 of 2,379 unclear links confirmed; 92 of 100 hand-checked right; two-publisher pairs 227 -> 280) |
| 3 | Early stop check on the 11 misses | seconds | go or stop. **Go (2026-11-03):** 4 of the 8 links held before each report's date; 5 of 8 with the model's links |
| 4 | Register the decisive test; build and freeze the 2025 set | listing minutes, download 1–2 hours | pre-registration file and a frozen set. **Done 2026-11-04:** registered with behaviour primary; frozen at 48 known and 69 open-set reports, below 60, so descriptive only |
| 5 | Both members, benchmarks sliced by date, tuning on the development sets, own combiner, floor and labels | combiner about 1 hour; benchmarks the usual hours | weights and model files. **Done (2026-11-04), no decisive run:** members, pins and artifacts built and tested, amendment 1 recorded before any run; no eligible weight pair beat weights 0 at the behaviour reading (22/90), so, as registered, step 6 is not run (amendment 2). A second attempt, store text through its own refitted combiner (amendment 3), put 38 vs 37 reports first at the behaviour reading but added one wrong name, so step 6 is again not run (amendment 4, 2026-11-05; technical documentation 12.9) |
| 5b | Dated store posts as combiner training lists (registered 2026-11-09, section above) | about 10–11 GPU hours planned, about 6 used | a combiner and a go or stop for step 6. **Done (2026-11-10), no decisive run:** the registered choice picked store weight 0, so no combiner was written |
| 6 | Decisive run, then the full gate (folds, CTI-HAL, Elastic) | hours | adopt or not |
| 7 | Later: artefacts and indicators; actors outside ATT&CK; dossier for the LLM | — | each with its own test |

## Costs

| Item | Cost |
|---|---|
| Reading reports with the local model | $0 on the laptop's GPU; about $0.62 on 2026-11-03 for a rented GPU when the laptop throttled |
| Optional second opinion on 200 links | about $1 of OpenRouter credit |
| Disk | a few GB during the backfill, a few hundred MB after |
| Daily refresh | a few minutes |

## Risks

| Risk | Mitigation |
|---|---|
| Extraction mistakes, such as a mention read as use | rules first, the model only for unclear cases, two publishers before a link can name, the 200-link check |
| Vendors disagree | keep both claims and treat the software as shared; an overlap never merges actors |
| Poisoning: a planted report names a false actor | approved sources only; two independent publishers to name; every fact links to its source |
| Sources change their layout or start refusing | per-source health in the status line; the store keeps working on the rest; refusals are never worked around |
| The gain shows in ranking but not in names | report both; the store never makes the naming gate easier to pass |
| The test set is too small | count it with a listing run before registering; the confirmation set keeps growing |

## Decisions for you

1. **Primary reading for the decisive test:** strict (malware names kept, where the store should
   help most) or behaviour (your choice for the last test). *Decided 2026-11-04: behaviour.*
2. **Backfill start:** 2022-01-01 (proposed) or earlier. *Decided: 2022-01-01 (step 1).*
3. **Sources:** the 33 approved sources as they are, or more. Each new source must be listable
   with a plain request. *Decided: the 33 as they are (step 1).*
4. **Actors outside ATT&CK:** may they ever be named (a later step), or only shown?

## Code map (proposed)

- `apt_classifier/knowledge_store/`
  - `sources.py`: the approved list, robots.txt, conditional requests.
  - `fetch.py`, `parse.py`, `names.py`, `extract.py`.
  - `store.py`: SQLite storage, `StoreView(as_of)` and a snapshot hash.
  - `members.py`: both members. It sits outside `classifiers/`, so at weight 0 the fingerprint is
    unchanged.
- `scripts/knowledge_store.py`: `backfill`, `update`, `extract`, `audit`, `status`,
  `view --as-of`.
- Make targets: `store-backfill`, `store-update`, `store-audit`.
- Settings:
  - `KNOWLEDGE_STORE_PATH` (default `data/knowledge_store/store.sqlite`, gitignored);
  - `STORE_SOFTWARE_WEIGHT=0` and `STORE_TEXT_WEIGHT=0`;
  - `STORE_MIN_PUBLISHERS_TO_NAME=2`.
- Benchmarks: `--store-as-of report` slices the store for each report. The fold harness hides
  cited posts and removed actors.
- Tests: mocked HTTP, an in-memory store, slicing by date, edge cases of name matching.
