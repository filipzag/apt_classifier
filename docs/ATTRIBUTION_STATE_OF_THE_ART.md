# APT attribution: the state of the art and what it means for Vestige

Research note, 2026-11-02. Sources are listed at the end; Vestige numbers come from the technical
documentation (sections 21.3.x) and are quoted as published there.

## Bottom line

- **What separates actors is their own malware and their specific habits, not the general
  technique list.** An independent study of ATT&CK (Saha, Lindorfer and Caballero, ASIA CCS 2026)
  found that only 34% of ATT&CK groups use any technique no other group uses, while 73% use
  software of their own. Vestige measured the same thing on real reports: the best technique-list
  model puts the right group first on at most 13–18% of real reports, while procedure text and
  named software carry the attribution.
- **The big published accuracy numbers (85–98%) do not transfer.** They come from closed answer
  lists, synthetic or generated reports, or set-ups where the answer leaks into the input (actor
  names in the text, group–technique links in the graph). The few tests on real reports with the
  names removed (CTIBench, AthenaBench, Vestige's fresh set) put the best systems at 39–63%
  first-choice accuracy.
- **Frontier language models win on current knowledge, not reasoning.** On AthenaBench, GPT-5
  went from 39% to 63% when it could search the web. Vestige's own head-to-head reached the same
  diagnosis: the model's wins came from malware and vendor reporting newer than ATT&CK.
- **Vestige is already at or ahead of practice on the hard part: not naming wrongly.** Its
  precision-first naming, evidence grading and leak-resistant testing match what the leading
  industry frameworks (Unit 42, Mandiant) and the newest research (abstaining classifiers, 2026)
  recommend.
- **The next gains come from data and linking, not from another classifier:** a dated knowledge
  store built from public reports, a "same activity as…" linking model for actors ATT&CK does not
  cover, and richer procedure artefacts. Training a local model makes sense only on top of that,
  and the useful thing to teach it is when not to name.

## 1. How attribution is done in practice

Practitioners work in layers. In interviews with 15 analysts (Saha et al., USENIX Security 2025)
the layers were: decide whether the activity is a targeted (APT) attack, tie it to known
tradecraft, and only then, rarely and with the most caution, attribute it to a country. Analysts
said they care more about *what* the adversary will do than *who* it is, and they want automation
that gives them a starting hint, not a verdict. For country attribution 12 of the 15 relied mainly
on infrastructure (IP ranges, hosting, registration data); some also used working hours, wording
and reused tools.

The industry frameworks formalise this:

- **Unit 42 Attribution Framework (July 2025).** Three levels: activity cluster (two or more
  related events), temporary threat group (one actor, observed for at least six months), named
  actor (all four Diamond Model facets, multiple reliable sources, estimative language). Every
  piece of evidence is graded with the Admiralty scale, with default credibility per artefact type
  (an IP address starts at 4 of 6 because it is volatile and shared). Level two asks for
  procedural detail beyond the general ATT&CK labels and for code similarity beyond hash matching.
- **Mandiant.** Uncategorised "UNC" clusters are merged into named groups only after review; since
  2018 a machine-learning model suggests the merges (see section 2.6).
- **The Q Model (Rid and Buchanan, 2015)** frames attribution as a graded judgement across
  technical, operational and strategic levels, not a yes/no fact.

### Evidence types, from most to least distinctive

| Evidence | How distinctive | How long it stays useful | Vestige today |
|---|---|---|---|
| Malware code lineage (shared code, not just hashes) | Highest; the industry's strongest technical link | Long; code is reused for years | Not analysed (Vestige reads reports and logs, not binaries) |
| Named software in the report | High; 73% of ATT&CK groups have their own, but only 24% in Malpedia | Long, until shared or leaked | Core signal ("software by rarity") |
| Procedure detail: exact commands, registry keys, file paths, configurations | High | Medium | Core signal (procedure text); statements naming a registry key pointed at the actor 77% of the time, commands and file paths 22%, plain descriptions 14% |
| Infrastructure: domains, IPs, certificates, hosting and registration patterns, TLS fingerprints | High for clustering, weak for naming | Short; rotates in weeks | IOCs accepted as input; no infrastructure history |
| Victims and strategic context (sector, region, motive) | Narrows, rarely decides | Long | Country/motivation members; narrows only |
| Timing and operator slips (working hours, language, typos, handles) | Supporting | Varies | Not modelled |
| ATT&CK technique list | Low; only 34% of groups have any unique technique | Long but shared by everyone | Counted once in the naming gate |

## 2. Automated methods in the literature

### 2.1 Technique-list matching and sequences

- **CAPTAIN** (Rani et al., 2024) orders a case's techniques along the kill chain and compares the
  sequence with each group's pattern: 61% first choice, 70% within two, on the authors' own
  technique dataset.
- **Technique Inference Engine** (MITRE CTID, 2024/2025) predicts techniques likely present but not
  yet observed, trained as a recommender on technique co-occurrence in reports.
- **Guru, Moss and Kochenderfer (2025)** used GPT-4 and embeddings to extract techniques and then
  trained an attribution model: the extracted lists differed a lot from human ones but still beat a
  baseline.
- **Counter-evidence.** "Kitten or Panda?" (2026) shows technique profiles rarely separate groups.
  "Synthetic APTs" (June 2026) had AI agents emulate APT28, APT29, APT41, APT44 and Lazarus in a
  test enterprise network: 55–80% of the techniques they used matched each group's ATT&CK profile,
  every agent opened with the same techniques whichever group it played, and in 8 of 10 runs the
  agent turned the defender's own monitoring tool into its command channel. Technique-level
  attribution is becoming forgeable.
- **Vestige.** The best technique-list model (a small neural network) puts the right group first
  on about 13% of CTI-HAL reports, and no model type got past 18%; the four technique-set members
  alone put it first on 0–1 of 55, and a fine-tuned language model reading the technique list on
  2–4%. Narrative order adds nothing beyond "techniques described near each other" (2026-11-02
  probe).

### 2.2 Graph models

- **APT-MMF** (Xiao et al., 2024): a graph of reports and their IOCs, text and attributes, with
  multi-level graph attention.
- **Tripartite graph network** (Mead and Arabo, Electronics 2025): groups, techniques and
  kill-chain stages with sentence embeddings; 85% accuracy on APTNotes. A reviewer flagged label
  leakage, since the group–technique edges already encode the answer.
- **CSKG4APT**: a multi-source knowledge graph matched against static group profiles.
- **Reading.** Graph models are mostly closed-set classifiers over the same profiles. The useful
  part for Vestige is the graph as a store of dated facts, not graph neural networks.

### 2.3 Text classifiers over reports

- Early work (NO-DOUBT 2019, Naveen et al. 2020, Irshad and Siddiqui 2023) trains classifiers on
  report text with high closed-set scores; actor names, aliases and malware names in the text make
  most of these numbers unreliable.
- **CyLens** (2025) reports 83–97% actor accuracy with 1B–70B models pre-trained on 271,570
  generated, vulnerability-centred reports; the test threats are not comparable to real APT
  reports.
- **Vestige.** A language model fine-tuned on anonymised ATT&CK sentences put the right actor first
  on 25.5% of real reports; a plain word-count (TF-IDF) document per actor got 49.1%. Exact words
  and names beat learned embeddings here.

### 2.4 Malware and binary attribution

- **Kaspersky Threat Attribution Engine:** "genotypes", short code fragments never seen in clean
  files, matched against 60,000+ targeted-attack samples; Kaspersky itself says the output cannot
  be used as evidence on its own.
- **ADAPT** (Saha et al., RAID 2024): links executables and documents into campaigns and groups,
  93–95% clustering precision; 6,134 samples from 92 groups.
- **DeepAPT** (2017): 98.6% on a two-country task; **VeriBin** (2024) asks "same author?" for
  binaries from unseen actors.
- **Williams, Sobey and Karafili (June 2026):** one yes/no classifier per group with an explicit
  "abstain"; with 87% of test samples from 60 groups never seen in training, it abstained on 94% of
  them and kept 92% precision. This is the same design choice as Vestige's naming gate.
- **Command lines:** Böge et al. (2024) attribute operators from the commands they run (many of
  them Cobalt Strike sessions), 95% on actors with many samples, lower on the rest.

### 2.5 Language models

| Benchmark | Set-up | Best published | Vestige |
|---|---|---|---|
| CTIBench threat-actor task (NeurIPS 2024) | Anonymised reports, free answer | GPT-4 52% right (86% "plausible") | 54% |
| AthenaBench threat-actor task (2025/26) | 100 masked report profiles, 25% non-ATT&CK actors | GPT-5 39%, with web search 63% | 23% |
| Vestige fresh head-to-head (2026-10-29) | 44 reports newer than the model, names masked | GPT-5.6 24/44 (behaviour only 18) | 19/44 (behaviour only 10) |

- **Retrieval helps most.** CTIArena (2025) finds closed-book models struggle and retrieval gives
  clear gains; AURA (2025) builds a multi-agent retrieval pipeline for attribution (no numbers
  published).
- **Reliability problems.** Mezzi, Massacci and Tuma (ARES 2025) found GPT-4o, Gemini 1.5 Pro and
  Mistral Large 2 inconsistent and overconfident on full-length reports; few-shot prompting and
  fine-tuning only partly helped. Meng et al. (2025) catalogue 15 failure types, among them
  "co-mention bias" (a report mentioning another actor), contradictory reports and unseen patterns.
  Vestige saw the same: a Black Basta mention read as use, and models naming far more often than
  they are right once malware names are masked (16 wrong of 34 names).

### 2.6 Linking instead of naming (industry)

Mandiant's model (deployed December 2018) represents each cluster as one document per "topic"
(malware, infrastructure, delivery, techniques, targets), scores each topic with TF-IDF cosine
similarity, and learns topic weights with a logistic regression. Training pairs come from fake
clusters made by sampling documents from the same established group. It suggests merges for
analysts to justify; Mandiant credits it with dozens of merges and notes it does not yet weigh
how recent the activity is.

The naming problem has its own fix: Microsoft and CrowdStrike published a shared mapping of their
names for more than 80 actors (June 2025), and the MISP galaxy maps vendor names to groups.

## 3. Where Vestige stands

**At or ahead of practice**

- Precision-first naming with abstention (naming gate, probability floor, "technique list counts
  once", "leader must beat the combiner's own none-of-these"), the design the newest malware
  research and Unit 42's level thresholds converge on.
- Procedure text as the attribution core, the "procedural detail beyond ATT&CK labels" Unit 42
  requires and the per-topic TF-IDF Mandiant uses.
- Admiralty-style evidence grading, ACH-style independent review, documented-overlap clusters.
- Evaluation stricter than almost every published paper: several masking levels, a fresh set newer
  than the models, a control for reports ATT&CK itself cites, pre-registration, open-set reports.

**Behind**

- Current knowledge. The model's lead on fresh reports came from malware lineage and actor habits
  reported after ATT&CK's snapshot; open structured lists (MISP, Malpedia) hold almost none of it
  (21.3.29).
- Actors ATT&CK does not list. A quarter of AthenaBench answers and much fresh vendor reporting
  name actors Vestige cannot name, so it can only abstain.
- No infrastructure history and no binary analysis (both need data sources Vestige does not have).

## 4. What to build next, and how

Ranked by expected gain per effort. Each needs the usual gate: a pre-registered test on reports
the change has never seen, with no new wrong names.

### 4.1 A dated knowledge store built from public reports (largest gain)

What it is: Vestige reads the public reports an organisation follows (vendor blogs, CISA and
national advisories) and keeps dated facts, each with its source link: "malware X attributed to
actor Y by vendor Z on date D", aliases, procedure snippets, sectors and regions targeted,
infrastructure. Facts and short quotes only, never the vendor's full text.

How:

1. Collect: RSS and advisory feeds (the Elastic and CISA plumbing exists), with per-source
   licence notes.
2. Extract: the local model splits each report into statements (the existing normaliser) and
   fills a fixed fact schema, the approach CTINexus (EuroS&P 2025) uses to build threat knowledge
   graphs from a few examples. "Mentioned" and "used" must be separate fields; that is the known
   Black Basta failure.
3. Resolve names: MISP galaxy (CC0/BSD) and vendor crosswalks map vendor names to ATT&CK groups;
   names with no ATT&CK group stay as their own actors.
4. Use: new malware-to-actor links feed the software member; procedure snippets extend the
   per-actor text documents (including non-ATT&CK actors); the language model gets the matching
   facts, with citations, as context.
5. Test with a time split: for each test report, only facts published before it count (the
   standard for drifting security data, Pendlebury et al. 2019). A new fresh set is needed; the
   current one is spent.

Risks: extraction errors, vendors disagreeing (keep per-vendor attributions with dates), and
deliberate poisoning through planted reports (weigh sources, require two vendors before a fact can
name).

Cost: local extraction is free on the GPU; with GPT-5.6 Sol it would be roughly the head-to-head
rate, about 6–7 cents a report.

### 4.2 The language model proposes, Vestige checks

Already registered (21.3.31: the model's names at stated confidence of at least 0.99, and the
switch rule, 29/44 first on the spent set). The store in 4.1 makes this stronger: the model then
argues from current, citable facts instead of memory, and Vestige can check each cited fact.

### 4.3 "Same activity as…" linking for actors ATT&CK does not list

When no ATT&CK group can be named, say "this matches vendor cluster X" or "this matches your
earlier case Y", the activity-cluster level of the Unit 42 framework.

The local activity-cluster store (`apt_classifier/activity_clusters.py`, opt-in) already keeps
case signatures over time, but its similarity is a hand-set blend that gives techniques the
largest weight (0.35, against 0.20 each for software and infrastructure, 0.15 timing, 0.10
victims) and plain overlap counts, so a common tool counts as much as custom malware. That is the
weakest evidence weighted highest.

How (the Mandiant recipe, on Vestige's evidence): one similarity per evidence topic (software,
procedure text and artefacts, infrastructure, targets, timing, techniques), each weighted by how
rare the shared items are; topic weights learned by a logistic regression from fake clusters
sampled from the same actor's documents (ATT&CK now, the store in 4.1 later). It runs on the CPU
in minutes and never needs an ATT&CK name to be useful.

### 4.4 Richer procedure artefacts

Extract exact artefacts as their own evidence topic: normalised command lines, registry keys, file
paths, mutex and pipe names, service and task names, user agents, debug (PDB) paths. Weight each
by how few actors share it. These are the details Unit 42 asks for, the ones that survive name
masking, and the ones AI agents emulating a profile do not reproduce.

### 4.5 Teaching a local model (after 4.1)

| Approach | Evidence | Verdict |
|---|---|---|
| Classifier from report text to a fixed actor list | Vestige 25.5% vs word counts 49.1%; published highs are closed-set | Do not repeat |
| Contrastive encoder: same actor across vendors as positives, actors sharing tools as hard negatives, names masked | Little published evidence; Vestige's embedding models lost to word counts | Small expected gain; keep the word-count channel beside it |
| Reasoning over retrieved facts, trained with a reward that punishes wrong names (+1 right, 0 abstain, a penalty for wrong, plus a calibration term) | RLCR (2025) cut confidence error from 0.37 to 0.03 with no accuracy loss; ByteDance (ICLR 2026) trained a 4B model to abstain | The most useful thing to teach: Vestige's naming discipline, inside the model |
| Distillation: a frontier model reasons over the same retrieved facts; a local model learns from its answers | Standard; cost known from the head-to-head | About $120–140 for 2,000 worked examples |

Two cautions. The model's knowledge should come from the store, not its weights, or it ages like
ATT&CK does. And training text is limited by licences: CTI-HAL has none, and vendor text cannot be
redistributed, so training must use ATT&CK-derived and licensed sets and keep vendor reports for
testing.

### 4.6 What not to do again

Both the literature and Vestige's results rule these out:

- More technique-list models (sets, sequences, graph networks over ATT&CK links, sequence models
  such as Mamba).
- Fine-tuning a classifier from report text to a fixed actor list.
- Comparing Vestige with published 85–98% figures: those are closed sets or leaky set-ups.
- A "none of these" option for actors removed from the knowledge base: the evidence supports the
  named group (21.3.23).

## 5. Adversarial outlook

- AI agents can now reproduce most of an actor's technique profile, so a perfect profile match with
  no specific artefacts deserves a warning rather than a name. Vestige's "technique list counts
  once" rule is the right base; a "profile-only match" flag is a cheap addition.
- Planted reports can poison any knowledge store, the one in 4.1 included: record each fact's
  source, date and reliability, and never let a single source name.

## Sources

Surveys, practice and frameworks

- Rani, Saha, Shukla. A Comprehensive Survey of Advanced Persistent Threat Attribution (2024). https://arxiv.org/abs/2409.11415
- Saha, Mattei, Blasco, Cavallaro, Votipka, Lindorfer. Expert Insights into Advanced Persistent Threats: Analysis, Attribution, and Challenges. USENIX Security 2025. https://www.usenix.org/conference/usenixsecurity25/presentation/saha
- Saha, Lindorfer, Caballero. Kitten or Panda? Measuring the Specificity of Threat Group Behaviors in Public CTI Knowledge Bases. ASIA CCS 2026. https://arxiv.org/abs/2506.10645
- Unit 42 Attribution Framework (2025). https://unit42.paloaltonetworks.com/unit-42-attribution-framework/
- Mandiant: Going ATOMIC, clustering and associating attacker activity at scale (2019). https://cloud.google.com/blog/topics/threat-intelligence/clustering-and-associating-attacker-activity-at-scale
- Mandiant: How Mandiant tracks uncategorized threat actors. https://cloud.google.com/blog/topics/threat-intelligence/how-mandiant-tracks-uncategorized-threat-actors
- Kaspersky Threat Attribution Engine. https://www.kaspersky.com/blog/kaspersky-threat-attribution-engine/35852/
- CrowdStrike and Microsoft threat-actor name mapping (2025). https://ir.crowdstrike.com/news-releases/news-release-details/crowdstrike-and-microsoft-collaborate-harmonize-cyber-threat
- Rid, Buchanan. Attributing Cyber Attacks. Journal of Strategic Studies 38 (2015). https://doi.org/10.1080/01402390.2014.977382

Methods

- Rani et al. Chasing the Shadows: TTPs in Action to Attribute Advanced Persistent Threats (CAPTAIN, 2024). https://arxiv.org/abs/2409.16400
- MITRE CTID. Technique Inference Engine. https://arxiv.org/abs/2503.04819 and https://ctid.mitre.org/projects/technique-inference-engine
- Guru, Moss, Kochenderfer. On Technique Identification and Threat-Actor Attribution using LLMs and Embedding Models (2025). https://arxiv.org/abs/2505.11547
- Xiao et al. APT-MMF (2024). https://arxiv.org/abs/2402.12743
- Mead, Arabo. APT Attribution Using Heterogeneous Graph Neural Networks with Contextual Threat Intelligence. Electronics 14(23) (2025). https://www.mdpi.com/2079-9292/14/23/4597
- Perry, Shapira, Puzis. NO-DOUBT (IEEE ISI 2019). https://archive.cps-vo.org/node/70174
- Naveen, Puzis, Angappan. Deep Learning for Threat Actor Attribution from Threat Reports (2020). https://cris.bgu.ac.il/en/publications/deep-learning-for-threat-actor-attribution-from-threat-reports/
- Irshad, Siddiqui. Cyber threat attribution using unstructured reports in CTI. Egyptian Informatics Journal (2023). https://doaj.org/article/43d11ae6d8cd44fc978f128f7c3702f6
- Liu, Liang et al. CyLens: LLMs as Threat Intelligence Copilots (2025). https://arxiv.org/abs/2502.20791
- Saha, Blasco, Cavallaro, Lindorfer. ADAPT it! (RAID 2024). https://github.com/SecPriv/adapt
- Rosenberg et al. DeepAPT (2017). https://arxiv.org/abs/1711.09666
- VeriBin, binary authorship verification for unknown actors (2024). https://doi.org/10.1145/3669901
- Williams, Sobey, Karafili. High-Precision APT Malware Attribution with Out-of-Scope Resilience (2026). https://arxiv.org/abs/2606.03523
- Böge, Ertan, Alptekin, Çetin. Unveiling Cyber Threat Actors: behaviour-based attribution (ACM DTRAP 2024). https://dl.acm.org/doi/full/10.1145/3676284
- Balassone et al. Synthetic APTs: the Collapse of TTP-Based Attribution (2026). https://arxiv.org/abs/2606.07158
- Cheng et al. CTINexus (EuroS&P 2025). https://arxiv.org/abs/2410.21060
- Rani, Shukla. AURA: multi-agent cyber threat attribution (2025). https://arxiv.org/abs/2506.10175

Language-model benchmarks and reliability

- Alam et al. CTIBench (NeurIPS 2024). https://arxiv.org/abs/2406.07599
- AthenaBench (2025, revised 2026). https://arxiv.org/abs/2511.01144
- CTIArena (2025). https://arxiv.org/abs/2510.11974
- Mezzi, Massacci, Tuma. Large Language Models Are Unreliable for Cyber Threat Intelligence (ARES 2025). https://arxiv.org/abs/2503.23175
- Meng et al. Uncovering Vulnerabilities of LLM-Assisted Cyber Threat Intelligence (2025). https://arxiv.org/abs/2509.23573
- CTI-HAL (WACCO 2025). https://arxiv.org/abs/2504.05866

Training and evaluation methods

- Damani et al. Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty (RLCR, 2025). https://arxiv.org/abs/2507.16806
- Wu et al. Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning (ICLR 2026). https://arxiv.org/abs/2512.19920
- Pendlebury et al. TESSERACT: Eliminating Experimental Bias in Malware Classification across Space and Time (USENIX Security 2019). https://www.usenix.org/conference/usenixsecurity19/presentation/pendlebury
- Arp et al. Dos and Don'ts of Machine Learning in Computer Security (USENIX Security 2022). https://www.usenix.org/conference/usenixsecurity22/presentation/arp
