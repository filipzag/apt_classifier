# Vestige — Explain It Like I'm Five

## What is this thing?

Imagine you're a detective. Someone broke into a house, and you need to figure out who did it. But you don't have one obvious clue — you have a big messy pile of clues: footprints, broken windows, a weird tool left behind, and some stories from neighbors.

This app is like a **detective's helper robot**. You give it all the clues from a computer break-in, and it helps you figure out which bad-guy group probably did it.

But here's the important part: **it doesn't say "THIS is the guy!"** It says "here are some guesses, ranked from most likely to least likely, and here's why I think so."

---

## The Big Picture

```
You give it clues → It reads the clues → It matches them to known bad guys → It ranks the suspects → You decide
```

That's it. Let's go through each step.

---

## Step 1: You Give It Clues

You can give the robot different kinds of clues:

- **Stories you tell it** — "I saw someone running PowerShell with a weird encoded command"
- **Alerts from security tools** — "The alarm went off at 3am"
- **Computer logs** — "Event ID 4688 shows process X started at time Y"
- **Known bad things** — "This IP address is evil" or "This file hash is bad"
- **Names of tools** — "They used Cobalt Strike"

The robot takes all of these and turns them into simple, clean sentences it can understand. Think of it like translating messy handwriting into neat printed text.

---

## Step 2: It Reads the Clues

The robot has a big library. The library is called **MITRE ATT&CK**. It's a giant catalog of all the bad things hackers do, organized like a dictionary.

For example, the library knows:
- **T1059.001** = "PowerShell" — using PowerShell to run commands
- **T1003.001** = "LSASS Memory" — stealing passwords from memory
- **T1486** = "Encrypted for Impact" — ransomware that locks your files

The robot takes your clues and **matches them to entries in this library**. It's like looking up "someone ran a weird PowerShell script" in the dictionary and finding "Ah, that's T1059.001."

It does this matching using **vectors** — which is just a fancy way of saying it turns words into numbers and compares how close the numbers are. Like how "dog" and "puppy" are close together, but "dog" and "spaceship" are far apart.

---

## Step 3: It Matches Clues to Known Bad Guys

The library also knows about **bad-guy groups** — like APT28, Lazarus, FIN7. Each group has a profile of which tricks they're known to use.

So if your clues show:
- PowerShell usage (T1059.001)
- LSASS memory access (T1003.001)
- A tool called "Cobalt Strike"

The robot checks: "Which groups are known to use these exact tricks?"

It's like: "Who in our criminal database is known for picking locks AND leaving behind a crowbar?"

---

## Step 4: It Ranks the Suspects

Here's the cool part. The robot doesn't just use one way to guess. It uses **many different judges**, like a panel at a talent show:

1. **The Pattern Judge** — checks which groups use the same techniques as your clues
2. **The Word Judge** — checks if your clue words match the group's description
3. **The Tool Judge** — checks if the tools you found match known group tools
4. **The Behavior Judge** — checks if the specific behaviors match how a group operates
5. **The Context Judge** — checks if the target, motive, and industry make sense for that group
6. **The Country Judge** — guesses which country the attackers probably work for (or none), then votes for the groups ATT&CK documents from there
7. **The AI Judge** — only when you ask for it, an AI (CyberRE-LLM) ranks the tricks behind one clue; its vote is small, it can't add a new suspect, and it never decides whether a group gets named

Each judge ranks the suspects independently. Then the robot combines all their rankings into one final list: half from adding up the judges' rankings, half from a learned combiner trained only on practice cases built from the library.

**Why so many judges?** Because no single judge is perfect. The Pattern Judge might be wrong if a technique is super common. The AI Judge might be biased. By using many judges and combining their votes, we get a more fair answer.

### The "Naive Bayes" Problem

One of the judges (Naive Bayes) used to be the boss. It would decide everything by itself. But that's dangerous — what if it's wrong? So now it's just one voice in the crowd. **No single judge gets to decide alone.**

---

## Step 5: The Interview

After the first ranking, the robot might ask you questions. Like a detective asking follow-up questions:

> "You said you saw PowerShell. Did you also see evidence of T1003.001 (LSASS access)? Check your Sysmon Event ID 10 logs for lsass.exe access."

These questions help the robot:
- **Find things you might have missed** — "Did you check for X?"
- **Confirm or deny hypotheses** — "If you didn't see X, that makes group Y less likely"
- **Recommend what logs to look at** — "Check Windows Security Event 4688 for process creation"

The robot knows which logs to recommend because it reads the **ATT&CK Detection Strategies** — these are like instruction manuals that say "to detect this trick, look at this specific log."

### The Telemetry Check

Here's something important: if the robot asks "Did you see X?" and you say "No," the robot checks: **"Did you actually have the right tools to see X?"**

If you don't have the right logs, your "No" doesn't count as evidence. It's like a detective asking "Did you see a footprint?" when you weren't even looking at the floor. A "No" only means something if you were actually looking.

---

## Step 6: The Attribution Stage

After all the analysis, the robot gives you one of four answers:

### 🟢 "We have a strong hypothesis"
There's enough evidence to say "Group X is our leading hypothesis." But it's still a **hypothesis**, not a confirmation. Like saying "I think Bob did it" — not "Bob definitely did it."

### 🔵 "We have a shortlist"
There are some candidates, but none of them is strong enough to be a hypothesis yet. Like saying "It could be Bob, Alice, or Charlie, but I'm not sure which one." The **Top three** view puts the first three side by side and shows which clues only one of them is known for.

### 🟠 "Unresolved activity cluster"
There's clearly bad stuff happening, but we can't match it to any known group. The robot gives it a label like "CL-UNK-1234" — like calling an unknown suspect "John Doe." This might be a new group we've never seen before.

### ⚪ "Insufficient evidence"
There's not enough evidence to say anything useful. Like trying to solve a case with just one blurry photo.

### What about percentages?

The robot gives **one** probability: the learned combiner's "leader probability", its estimate that the top suspect is the right group (shown when the combiner is on, which is the default). It was fitted on practice cases built from the ATT&CK library, not on real casework, so a "78%" there is a rough guide, not a real-world chance that Group X did it. The stage itself stays qualitative.

---

## Step 7: The Review

Before the final answer, the robot runs an **independent AI review**. This is separate from the main analysis — like having a second detective review the first detective's work.

The review has several stages:

1. **Blinded review** — the AI sees the evidence but NOT the suspect names. It ranks them as "Candidate A," "Candidate B," etc. This prevents fame bias (giving more weight to famous groups).
2. **Unblinded review** — now the AI sees the real names and can use its knowledge.
3. **Fame-bias check** — compares blinded vs. unblinded to see if the AI was biased by famous names.
4. **Devil's advocate** — the AI argues AGAINST the leading hypothesis to find weaknesses.
5. **Reconciliation** — the AI combines everything into a final advisory opinion.

**But here's the key:** the AI review is advisory only. It cannot override the deterministic analysis. The robot's main ranking is the boss; the AI review just gives advice.

---

## Step 8: Live Threat Intelligence

The robot can also check **live threat feeds** — like CISA alerts and ransomware trackers. But these feeds are **non-voting**. They can add context ("Hey, there's a recent alert about this technique") but they can't vote in the ranking. They're like a tip line — useful information, but not a vote in the jury.

---

## Step 9: The Canvas and Diamond Model

The robot gives you two visual tools:

### Analyst Canvas
A big map showing how everything connects:
- Your evidence → links to → ATT&CK techniques → links to → suspect groups
- Tools you found → links to → groups known to use them
- Campaigns → links to → groups

It's like a detective's corkboard with strings connecting all the clues.

### Diamond Model
Organizes the intrusion around four corners:
- **Adversary** — who did it
- **Capability** — what tools/tricks they used
- **Infrastructure** — what systems they used
- **Victim** — who was attacked

---

## Step 10: Exports

You can download several things:

- **JSON audit** — everything the robot found, in machine-readable format
- **Markdown report** — a human-readable report
- **ATT&CK Navigator layers** — a visual overlay showing which techniques were observed, which were checked and absent, and where you have telemetry gaps
- **Attack Flow (STIX)** — a step-by-step timeline of the attack, exportable to other security tools

---

## The Golden Rules

Here's what makes this robot different from a regular AI:

1. **It never says "THIS is the guy."** It always says "here's my best guess, and here's why."
2. **No single judge decides.** Multiple independent classifiers vote, and no one of them is the boss.
3. **The AI review is separate.** The AI can give advice but cannot override the deterministic analysis.
4. **Live intelligence can't vote.** It adds context but doesn't get a say in the ranking.
5. **No made-up percentages.** The one probability shown comes from the learned combiner, which says what it was fitted on; the stages themselves are qualitative.
6. **What goes to the cloud, and what does not.** Windows and Sysmon logs are parsed on your own computer.
   With an OpenRouter key, the online AI service receives the written clues, alert text, interview context, the
   observations the technique checker double-checks (these can include text taken from logs, such as command
   lines) and the AI review. Without a key (local mode) nothing is sent.
7. **"No" only counts if you were looking.** A negative answer is only evidence if you had the right tools to see it.
8. **Uncertainty is preserved.** The robot would rather say "I don't know" than guess wrong.

---

## Simple Analogy

Think of the whole system like this:

| Real world | This app |
|---|---|
| Crime scene | Computer incident |
| Clues | Logs, alerts, observations, IOCs |
| Criminal database | MITRE ATT&CK knowledge base |
| Suspects | Threat actor groups (APT28, Lazarus, etc.) |
| Detectives | Multiple classifiers (judges) |
| Detective's panel vote | Reciprocal Rank Fusion |
| Second detective review | AI review (blinded, unblinded, devil's advocate) |
| Tip line | Live threat intelligence (non-voting) |
| Case file | JSON audit export |
| Evidence board | Analyst Canvas |
| Investigation timeline | Attack Flow |

---

## What This App Is NOT

- It's **not** a lie detector. It can't prove who did it.
- It's **not** an AI that decides everything. The AI is just one small voice.
- It's **not** magic. It can only match what you give it to what's in the library.
- It's **not** always right. Shared tools, incomplete reporting, and false flags can fool it.
- It's **not** a replacement for a human analyst. It's a helper.

---

## Glossary (for when you're ready to grow up)

- **APT** — Advanced Persistent Threat. A fancy word for "a skilled bad-guy group that sticks around."
- **ATT&CK** — A big dictionary of bad things hackers do, made by MITRE.
- **Attribution** — Figuring out who did it.
- **Classifier** — A judge that ranks suspects.
- **RRF** — Reciprocal Rank Fusion. A way to combine multiple judges' rankings into one.
- **ChromaDB** — A local database that stores vectors (numbers representing words).
- **Vector** — A list of numbers that represents the meaning of a word or sentence.
- **STIX** — A standard format for sharing threat information between security tools.
- **Telemetry** — The logs and sensors you have that can see what's happening on your computers.
- **Calibration** — Checking how often the robot was right in the past, so it can give real probabilities.
