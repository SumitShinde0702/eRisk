# eRisk 2026 Task 1 — Research narrative for paper generation

This document summarizes **what was built and evaluated** for the CLEF **eRisk 2026 Task 1** (“Conversational Depression Detection with LLM Personas”). It is intended as **source material** for another system to draft a research paper (methods, system description, ablations, limitations).

**Official task page:** [eRisk 2026 Task 1 (LLMs)](https://erisk.irlab.org/Task1LLMs.html)

---

## 1. Task and objective

**Setting:** An **agent-to-agent (A2A)** pipeline where a **doctor agent** conducts a natural-language interview with a **simulated patient persona**. The system must infer **Beck Depression Inventory–II (BDI-II)**–style evidence **indirectly** (no blunt “Are you depressed?” screening as the default strategy) and produce **submission JSON**: per-persona conversation logs and predicted **total BDI score** plus up to **four key symptoms**.

**Competition constraints (high level):**

- **20 personas** (IDs 1–20), each with a dedicated LoRA adapter on a shared base LLM.
- **Three official runs** (run 1–3) with distinct hyperparameter profiles (see §7).
- Outputs uploaded via FTP under a prescribed folder layout (`task1-llms-results/personaN/`).
- At most **one** “manual” run (human-assisted) across the submission, indicated by filename prefix `manual_`.

---

## 2. Scientific / engineering goals

1. **Indirect probing:** Use conversational, topic-guided questions aligned to BDI domains without naming depression as the opening move.
2. **Structured inference:** Map free-text dialogue to **21 BDI-II symptom dimensions** scored **0–3**, then aggregate to a **0–63** total consistent with BDI-II scoring conventions.
3. **Safety-aware dialogue:** When **acute self-harm / hopelessness** cues appear, prioritize **clarification and graded safety probing** (an “acute ladder”) before stopping.
4. **Controllable interview length:** Balance **recall** (enough coverage) vs **precision** (avoid over-probing “control” personas) via rules + run policies.
5. **Reproducibility:** Separate **API-based** components (DeepSeek for prober/extractor) from **deterministic** rules (stopper/scorer) and **optional** embedding-based retrieval.

---

## 3. System architecture (multi-agent)

The implementation lives primarily under `src/`. The **orchestrator** (`src/orchestrator.py`) runs a turn loop:

| Component | Role | Implementation notes |
|-----------|------|----------------------|
| **Stopper** | Continue vs **CLASSIFY** | Rule-based (`src/agents/stopper.py`): min depth, group-screen coverage, core-domain coverage, acute-risk gating, early-stop heuristics. |
| **Prober** | Next doctor question | LLM (**DeepSeek** via OpenAI-compatible API, `deepseek-chat`) with a fixed system prompt emphasizing indirect, topic-linked follow-ups and **red-flag** follow-up before topic switches (`src/agents/prober.py`). Integrates **cluster routing**, **topic hierarchy**, **YAML interview banks**, and optional **evidence retrieval** from prior patient turns (`src/agents/evidence_memory.py`). |
| **Persona (patient)** | Simulated reply | **Real:** `meta-llama/Meta-Llama-3-8B-Instruct` + per-persona LoRA (`Anxo/erisk26-task1-patient-{:02d}-adapter`). **Mock:** keyword- or AI-driven stubs for dev (`src/persona_client.py`). Mandatory **frozen** system prompt per task rules. |
| **Template / risk evidence** | Turn-level risk features | Matches patient replies against **symptom templates** and a **risk lexicon**; can use **sentence-transformer embeddings** when enabled (`src/agents/template_evidence.py`, `knowledge/symptom_templates.yaml`, `knowledge/risk_lexicon.yaml`). |
| **Extractor** | Symptom signals from dialogue | LLM (**DeepSeek**) outputs JSON over the **21 exact BDI symptom strings** with **0–3** scores; merges across turns by taking stronger evidence (`src/agents/extractor.py`). Includes detailed **suicide / fatalism / paradoxical “positive” framing** instructions to reduce under-scoring. |
| **Scorer** | Final BDI total + key symptoms | Sums symptom scores (clamped), applies **acute-risk calibration** that can raise totals when suicidal ideation + ladder coverage + buffer risk align (`src/agents/scorer.py`). Selects up to **four** highest non-zero symptoms (ties broken in BDI order via `validate_key_symptoms` in `src/bdi_mapper.py`). |

**Risk routing:** `src/agents/risk_router.py` defines **clusters** (e.g. `AcuteSafety`, `HopelessWorthless`, …), **acute euphemism** patterns, and a structured **acute ladder** (intent → plan → timeline → means → protective factors) used both for **question selection** and **stop gating**.

---

## 4. Conversation mechanics (what makes it “not a single LLM chat”)

1. **Topic hierarchy and symptom groups:** `src/topic_hierarchy.py` organizes probing so questions flow **general → specific** across mood, vegetative, cognitive, and behavioral buckets.
2. **YAML interview banks:** `src/agents/interview_banks.py` + `knowledge/group_screen_questions.yaml`, `knowledge/symptom_drilldown_questions.yaml` provide **screening** and **drill-down** questions per group/symptom, with caps (`max_questions_per_symptom`, `max_questions_per_group`).
3. **Bank follow-up:** After a bank screen/drilldown question, an **extra LLM follow-up** (`get_bank_followup_question`) anchors on the patient’s last answer (`bank_followup_enabled` in policy).
4. **Evidence memory:** The prober can retrieve **top-k paraphrases** of relevant past patient statements to ground the next question (`retrieve_relevant_patient_evidence`); embeddings can be disabled for lightweight runs.
5. **Risk buffer:** Each assistant turn gets a **template-based risk score**; the buffer keeps top risky turns with a **recency-weighted** blend for **acute calibration** and **stop logic**.
6. **Stopping policy highlights** (`should_stop`):
   - Enforce **minimum exchanges** and **group screen** completeness (each functional group must receive enough screen questions).
   - Require **minimum distinct group coverage** before early exit.
   - If **acute risk** is present, delay stopping until **minimum depth** and **acute ladder progress** thresholds are met (policy: `required_acute_ladder_steps`).
   - **Positive framing early stop:** if early patient turns sound globally well, use a **relaxed** total-score threshold to avoid over-probing non-depressed-like trajectories.

---

## 4A. Deep dive: clusters, question banks, templates, and rules (with examples)

This section lists **representative content actually used in the codebase** so a paper can quote **concrete probes**, **routing logic**, and **lexical features**. Source files are cited in parentheses.

### 4A.1 Six routing clusters (`src/agents/risk_router.py`)

The prober’s **risk router** classifies the *current* dominant **interview cluster**. Allowed values: `AcuteSafety`, `HopelessWorthless`, `CoreDepression`, `VegetativeCognitive`, `BehavioralArousal`, `GeneralCheckin`.

**Classifier prioritization (paraphrased from code):**

- If the patient indicates **imminent self-harm, intent, plan, or “doing it soon”** → **AcuteSafety**.
- If **resigned / fatalistic / worthlessness** language dominates **without** clear imminence → **HopelessWorthless**.
- Otherwise pick the best **non-risk** cluster (`CoreDepression`, `VegetativeCognitive`, `BehavioralArousal`) or **GeneralCheckin**.

**If no DeepSeek API key:** routing falls back to **lexical pattern matching** over recent patient text (and optionally the **risk buffer**), checking clusters in priority order: acute euphemisms → `AcuteSafety` lexical list → other cluster keyword lists → `GeneralCheckin`.

**Example questions drawn from `CLUSTER_QUESTION_BANK` (one line each cluster):**

| Cluster | Example question (verbatim from code) |
|---------|----------------------------------------|
| **AcuteSafety** | “You said this might happen soon - what has made things feel this immediate right now?” |
| **HopelessWorthless** | “Do you feel like people would be better off without you, or is it more that you feel stuck?” |
| **CoreDepression** | “What used to lift your mood before, and does any of that still help now?” |
| **VegetativeCognitive** | “How has your sleep been affecting your day-to-day energy lately?” |
| **BehavioralArousal** | “Have you felt more on edge or restless than usual?” |
| **GeneralCheckin** | “What has been the hardest part of your days recently?” |

Additional cluster-bank lines in code include, for example: “How alone do you feel with this right now?” (AcuteSafety); “How do you see the next few weeks unfolding for you?” (HopelessWorthless); “When during the day do you feel the worst?” (CoreDepression); “Do simple decisions feel harder than they used to?” (VegetativeCognitive); “When stress rises, what does it look like in your body or behavior?” (BehavioralArousal).

### 4A.2 Acute safety ladder (graded follow-up)

When **acute safety cues** are present, the system tracks progress through **`ACUTE_LADDER`** — a sequence of **stages**, each with a **canonical question** and **substring markers** used to detect whether that stage was already asked (for `acute_ladder_progress` and `next_acute_ladder_question`).

**Stages (intent → plan → timeline → means → protective factors):**

1. **intent_clarification:** “When you say this will end soon, are you talking about ending your life?” (markers include “ending your life”, “talking about ending”, “end your life”).
2. **plan_concreteness:** “Have you thought about specific steps or a specific way you would do it?” (markers: “specific steps”, “specific way”, “how you would do it”, “plan”).
3. **timeline_imminence:** “Have you thought about when you might do this?” (markers: “when you might”, “timeline”, “soon”, …).
4. **means_access:** “Do you currently have access to what you would use?” (markers: “access”, “means”, …).
5. **protective_factors:** “What has stopped you so far from acting on these thoughts?” (markers: “stopped you so far”, “acting on these thoughts”, “kept you safe”, …).

**Acute euphemism examples** (non-exhaustive) matched before general lexical rules: “end soon”, “all going to end”, “leave earth”, “won’t be here”, “waiting for the timer”, “no one would notice if i wasn’t here”, “want to disappear”, …

**Ambiguous acute cues** (code comment: require **two** hits to contribute to acute signal): e.g. “disappear”, “just existing”, “end of something”, “waiting for the timer”.

### 4A.3 Lexical routing cues (offline / no-LLM cluster)

`_LEXICAL_RULES` maps clusters to **phrase lists** (patient text, lowercased). Examples:

- **AcuteSafety:** “end it”, “kill myself”, “going to end”, “die”, “gone soon”, “leave the earth”, “suicide”, “don’t want to be here”, …
- **HopelessWorthless:** “nothing matters”, “pointless”, “worthless”, “better without me”, “accepted my fate”, “know how this ends”, “made my peace”, “hopeless”, …
- **CoreDepression:** “down”, “sad”, “empty”, “don’t enjoy”, “anhedonia”, “no interest”
- **VegetativeCognitive:** “sleep”, “insomnia”, “tired”, “fatigue”, “focus”, “concentrate”, “appetite”
- **BehavioralArousal:** “restless”, “agitated”, “irritable”, “on edge”, “snappy”

### 4A.4 Four functional groups and YAML **screen** questions (`knowledge/group_screen_questions.yaml`)

Group **order** for screening: **Affective → Executive → Somatic → Cognitive**. Policy requires **at least `min_questions_per_group` screen questions per group** (default **3**) before certain early stops; the orchestrator picks the **next** group with the **fewest** screen questions so far.

**Examples of **screen** question text and **BDI symptom targets** (as in YAML):**

- **Affective:** “How have you been feeling lately, overall?” → targets `Sadness`, `Loss of Pleasure`. “When you think about the next few weeks, what do you expect day to day?” → `Pessimism`, `Sadness`. “How do you feel about yourself these days compared with how you used to?” → `Self-Dislike`, `Self-Criticalness`.
- **Executive:** “Is it harder to focus or concentrate than it used to be?” → `Concentration Difficulty`. “Do you find it harder to make decisions than before, even small ones?” → `Indecisiveness`. “How do you feel about how things have gone for you recently?” → `Past Failure`, `Guilty Feelings`.
- **Somatic:** “How have you been sleeping lately?” → `Changes in Sleeping Pattern`. “Do you feel like you have enough energy for your usual day?” → `Loss of Energy`, `Tiredness or Fatigue`. “Have you been more irritable or short-tempered than usual?” → `Irritability`, `Agitation`.
- **Cognitive:** “Been doing anything you enjoy lately?” → `Loss of Pleasure`, `Loss of Interest`. “Are you still into the things you used to care about?” → `Loss of Interest`, `Loss of Pleasure`. “Have you noticed any changes in what you're interested in lately?” → `Loss of Interest in Sex`, `Loss of Interest`.

These questions are **indirect** (no “depression” label in the question text by design).

### 4A.5 Symptom **drilldown** bank (`knowledge/symptom_drilldown_questions.yaml`)

After screening, the system can ask **narrowing** follow-ups per symptom (up to `max_questions_per_symptom`, default **2**), prioritized by **current extractor signal strength** then **BDI order**. Examples:

- **Sadness:** “When you say things have been rough, is that most days or just now and then?” / “Does anything briefly lift your mood, or does it stay flat?”
- **Suicidal Thoughts or Wishes:** “When things feel really heavy, what thoughts show up about getting through it?” / “Have you had moments of wishing you weren't here, even if you didn't plan anything?”
- **Worthlessness:** “Do you feel like you matter to people around you right now?” / “When you're quiet, what story about yourself runs in the background?”
- **Changes in Sleeping Pattern:** “Is the problem falling asleep, staying asleep, waking too early, or sleeping too much?” / “How many nights a week is it like that?”
- **Concentration Difficulty:** “When you read or work, how often do you have to re-read or restart?” / “Is it worse at certain times of day?”

### 4A.6 Topic hierarchy (`src/topic_hierarchy.py`) — topics, keywords, and example probes

Six **`Topic`** records define **keywords**, **mapped BDI symptoms**, **opening_questions**, **follow_up_questions**, and **related_topics** for coherent multi-turn probing.

**Examples:**

- **General_Mood** — keywords: “feel”, “feeling”, “mood”, “sad”, “down”, “low”, “okay”, “fine”, … — symptoms: `Sadness`, `Loss of Pleasure`. Opening: “How have things been for you lately?” Follow-up: “When you say that, what does a typical day feel like for you?”
- **Physical** — keywords: “sleep”, “tired”, “fatigue”, “energy”, “appetite”, “insomnia”, … — symptoms include sleep, fatigue, energy, appetite. Opening: “How have you been sleeping?” Follow-up: “You mentioned sleep - has that affected your energy during the day?”
- **Motivation** — keywords: “interest”, “enjoy”, “hobbies”, “pointless”, … — symptoms: `Loss of Interest`, `Loss of Pleasure`, `Loss of Interest in Sex`. Opening: “Been doing anything you enjoy lately?” Follow-up: “When you try to do those things, how does it feel?”
- **Self_Outlook** — keywords: “worth”, “failure”, “guilty”, “hopeless”, “useful”, … — seven symptoms including `Pessimism`, `Worthlessness`, guilt/self-blame items. Opening: “How do you see things going for you in the near future?” Follow-up: “Do you feel like you're still useful or needed?”
- **Cognitive** — keywords: “focus”, “concentrate”, “decide”, … — `Indecisiveness`, `Concentration Difficulty`. Opening: “Is it harder to focus or concentrate than it used to be?”
- **Behavioral_Emotional** — keywords: “cry”, “irritable”, “restless”, … — `Crying`, `Agitation`, `Irritability`. Opening: “Have you found yourself getting emotional more easily lately?”

**Four-way symptom grouping** (`SYMPTOM_GROUPS`: Affective, Executive, Somatic, Cognitive) maps each BDI item to exactly one group for **coverage balancing** (e.g. `Loss of Interest` → Cognitive; `Past Failure` → Executive; `Agitation` → Somatic).

### 4A.7 BDI-aligned **indirect** question bank (`src/bdi_mapper.py` — `BDI_QUESTION_BANK`)

The codebase maintains **one canonical indirect question per BDI item** (same order as `BDI_SYMPTOMS`). Examples:

- Sadness: “How have you been feeling lately?”
- Pessimism: “How do you see things going for you in the near future?”
- Loss of Pleasure: “Been doing anything you enjoy recently?”
- Suicidal (item 9): **no direct suicide ask** — placeholder: “What's been on your mind when you're alone?”
- Changes in Sleeping Pattern: “How have you been sleeping lately?”
- Concentration Difficulty: “Is it harder to focus or concentrate than it used to be?”

**High-yield symptom indices** probed first in some paths: indices `0, 11, 15, 19, 14` → `Sadness`, `Loss of Interest`, `Changes in Sleeping Pattern`, `Tiredness or Fatigue`, `Loss of Energy`.

### 4A.8 Prober LLM **system** rules (`src/agents/prober.py`)

The prober’s system prompt instructs the model to output **only one** natural question; **never** ask “Are you depressed?” or use the word “depression” / “mental health” in disallowed ways; move **general → specific**; **build on the patient’s last message**; and **if the patient says something concerning or ambiguous**, ask a **follow-up before changing topic** (with explicit examples: “accepted my fate”, “know how this ends”, “made my peace”, “it doesn’t matter anymore”, “how it ends”).

**Red-flag substring patterns** (non-exhaustive): “accepted my fate”, “know how this ends”, “end it”, “better without me”, “not wanting to be here”, “want to die”, “wanna die”.

**Bank follow-up** (`get_bank_followup_question`): after a YAML screen/drilldown question, a **second** LLM turn generates one short follow-up grounded in the patient’s answer; if the API is missing, **fallback strings** include e.g. “What part of that has been the hardest for you day to day?” and “When did you first notice that?”

### 4A.9 Symptom **templates** for turn scoring (`knowledge/symptom_templates.yaml`)

Templates are short **first-person** statements used for **embedding similarity** (or lexical overlap) to label a patient turn with likely symptoms. Examples:

- `Sadness`: “I feel sad.” / “I feel down most days.”
- `Pessimism`: “I am discouraged about my future.” / “The future feels hopeless.”
- `Suicidal Thoughts or Wishes`: “I have thoughts of killing myself.” / “I am going to end it soon.” / “I will leave earth soon.”
- `Worthlessness`: “I feel worthless.” / “People would be better without me.”
- `Changes in Sleeping Pattern`: (in file) insomnia/hypersomnia-related lines.

### 4A.10 Risk **lexicon** and composite **turn risk score** (`knowledge/risk_lexicon.yaml`, `compute_turn_risk_score`)

`acute_safety` phrases include: “going to do it”, “end it”, “kill myself”, “want to die”, “wanna die”, “do not want to be here”, “leave earth”, “all going to end soon”, “it will all be over”, …  
`hopeless_worthless` includes: “nothing matters”, “pointless”, “accepted my fate”, “know how this ends”, “made my peace”, “better without me”, “worthless”, “no point”, …

**Scoring:** `compute_turn_risk_score` starts from the **max template match score**, adds **+0.5** if any acute lexicon hit, **+0.2** if hopeless lexicon hit, **+0.2** if a matched template symptom is `Suicidal Thoughts or Wishes`, then **clips** to **1.0**.

### 4A.11 Evidence memory (`src/agents/evidence_memory.py`)

Patient-side retrieval uses **`sentence-transformers/all-MiniLM-L6-v2`** embeddings and **cosine similarity**; returns top patient utterances with similarity **> 0.2**. If the embedder is disabled or unavailable, **token-overlap** ranking is used (same idea as the ablation script’s offline memory).

### 4A.12 Extractor: instructions and **post-processing** (`src/agents/extractor.py`)

The extractor system prompt requires **JSON** with **exact** 21 BDI symptom keys and scores 0–3, with long **CRITICAL** blocks on: suicidal wording; **explicit** plans; **fatalistic** language (“accepted my fate”, “ik how this ends”) even with paradoxical “positive” tone; and **not** treating genuine positive overall health reports as severe somatic symptoms.

**Post-parse rules** include: **hopeful** patient markers (“i hope things get better”, “i’m trying”, …) **without** acute phrases → cap `Pessimism`, `Worthlessness`, `Suicidal Thoughts or Wishes` at **1** if the model scored higher; **evidence-count gating** for some symptoms (e.g. `Agitation`, `Indecisiveness`, `Worthlessness`, `Concentration Difficulty`) — if score > 1 but fewer than **two** patient messages match domain patterns, **clamp to 1**; **safety override** phrases (“gona end it”, “sleep forever”, “happy to die”, …) force `Suicidal Thoughts or Wishes` ≥ **3**.

### 4A.13 Fallback extractor **keyword** patterns (`extract_symptoms_fallback`)

When no API is available, a **fixed ordered list** maps keyword groups to **symptom indices** (Sadness, Pessimism, …): e.g. mood words → Sadness; “hopeless”, “accepted my fate”, “know how it ends” → Pessimism; self-harm language → Suicidal; “worthless”, “better without me” → multiple relevant items. This is **not** the primary production path when `DEEPSEEK_API_KEY` is set.

### 4A.14 Scorer: acute **calibration** (conceptual)

`score_bdi` sums symptom scores, then `_calibrate_score_for_acute_risk` may **raise** the total when: `has_acute_signal` is true, **suicidal** symptom signal is high, **`acute_ladder_progress`** meets `required_acute_ladder_steps`, and **risk_buffer** max template risk exceeds thresholds — using run-policy floors like `acute_boost_floor`, `moderate_acute_boost_floor`, `mild_acute_boost_floor`.

### 4A.15 Stopper: **positive framing** and **core coverage** (`src/agents/stopper.py`)

**Early positive framing** phrases include: “doing well”, “feeling good”, “pretty good”, “alright”, “doing okay”, “i’m good”, “nothing to report”, … If these appear in the **first few assistant messages**, stopping can trigger sooner with a **higher** allowed sum threshold (`positive_framing_threshold`) to reduce over-probing.

**Core domain coverage** before early stop requires extractor signal (or interview coverage) spanning **sleep/energy**, **interest/pleasure**, **self-view** (worthlessness/self-criticism cluster), and **future** (pessimism or user questions mentioning future/next weeks).

### 4A.16 Mock persona **keyword routing** (`src/persona_client.py`)

In **keyword mock** mode, user questions are matched to **domains** (sleep, interest, energy, appetite, mood, …) and mapped to **tiered** scripted replies for personas **1–8** (e.g. suicidal through “happy” tiers). Comments in code note that **broad** keywords like “lately” are placed **late** in the map so they do not **collapse** variety across probes.

### 4A.17 Extended rubric (`newVersion15Mar/config/symptom_rubric.yaml`)

The March 2025 rubric adds **per-symptom** definitions, **positive/negative probes**, **common_confounders**, and **severity_cues** (mild/moderate/severe), plus **risk_tags** (e.g. `SUICIDE`, `INSOMNIA`) with **trigger_examples** for annotation and development (e.g. suicide tag: “it will all end up soon”, “sleep forever”; eating disorder confounders for appetite-related items).

---

## 5. Knowledge artifacts (curated content)

| Asset | Purpose |
|-------|---------|
| `knowledge/run_policies.yaml` | Per-run **hyperparameters** (thresholds, caps, temperatures, acute floors). |
| `knowledge/symptom_templates.yaml` | Paraphrases for **template matching** / embedding similarity. |
| `knowledge/risk_lexicon.yaml` | Phrase lists for **acute** vs **hopeless/worthless** cues. |
| `knowledge/risk_ladder.yaml` | Structured ladder content complementary to code (`ACUTE_LADDER`). |
| `knowledge/group_screen_questions.yaml`, `knowledge/symptom_drilldown_questions.yaml` | Structured bank questions. |
| `knowledge/talkdep_golden_truth.yaml` | **External** reference BDI scores for **TalkDep** personas — **not** eRisk’s hidden labels for personas 1–20. |
| `knowledge/cluster_playbooks.md` | Human-readable **cluster objectives** for safety vs depression vs vegetative probing. |
| `newVersion15Mar/config/symptom_rubric.yaml` | Extended **symptom definitions**, example probes, confounders (March 2025 professor-note track). |

---

## 6. Models and APIs

- **Doctor-side LLM:** DeepSeek **`deepseek-chat`** at `https://api.deepseek.com` for **Prober** and **Extractor** (and optional **AI mock persona** replies when `--mock` without `--keyword-mock`).
- **Patient-side LLM:** Hugging Face **Llama-3-8B-Instruct** + **per-persona LoRA** adapters (gated; requires `HF_TOKEN`).
- **Embeddings (optional):** `sentence_transformers` for template similarity and/or memory retrieval; can be turned off via env flags such as `DISABLE_TEMPLATE_EMBEDDINGS=1`.

Environment variables (see `.env.example` in repo): `DEEPSEEK_API_KEY`, `HF_TOKEN`, and optional tuning like `MAX_MESSAGES`, `MIN_EXCHANGES_BEFORE_STOP`.

---

## 7. Official run policies (three submission profiles)

Defined in `knowledge/run_policies.yaml`:

| Run | Name (intent) | Sketch of differences |
|-----|----------------|------------------------|
| **run1** | `balanced` | Moderate **severe_threshold** (24), **acute_boost_floor** 50, default min exchanges 10, ladder steps 3. |
| **run2** | `high_recall` | Lower **severe_threshold** (20), **fewer** ladder steps required (2), **more permissive** control threshold (6), slightly **shorter** min exchanges (9) — tilts toward not missing depression/safety signals. |
| **run3** | `high_precision` | Higher **severe_threshold** (28), **stricter** control threshold (4), **more** ladder steps (4), **longer** min exchanges (11), higher acute floors — tilts toward fewer false positives / more confirmation. |

These names are **engineering labels**; the paper should describe them as **recall-oriented vs precision-oriented** profiles realized through **threshold and stopping** differences, not as separate models.

---

## 8. Outputs and submission format

CLI entry: `run.py`.

For each run ID, the pipeline writes (by default under `outputs/`):

- `interactions_run{N}.json` — list of `{ "LLM": persona id, "conversation": [...] }`.
- `results_run{N}.json` — list of `{ "LLM": persona id, "bdi-score": int, "key-symptoms": [...] }`.

**FTP upload** mirrors local `task1-llms-results/personaN/` structure (see root `README.md` for `lftp` examples).

---

## 9. Evaluation and auxiliary experiments (not official eRisk test labels)

### 9.1 TalkDep external corpus

The repo includes **TalkDep**-derived **final conversations** and a **golden ranking** file for **development calibration**:

- `scripts/run_talkdep_eval.py` — parses patient lines, runs extractor scoring, reports metrics (e.g. Spearman / MAE as implemented).
- `scripts/eval_talkdep_ranking.py` — ranking agreement when results use TalkDep persona names.

**Important caveat for any paper:** TalkDep references are **external**; they do **not** substitute for the official eRisk 1–20 evaluation.

### 9.2 Submission summarization

- `scripts/eval_submission_summary.py` — aggregates existing `results_run*.json` under a folder tree (e.g. `outputs/submission`).

### 9.3 Isolated experiment workspace: `newVersion15Mar/`

Created to run **follow-up analyses** without destabilizing the main submission pipeline.

**Component ablation (`scripts/run_component_ablation.py`):**

- **Variants:** `baseline`, `no_memory`, `no_template_risk`, `no_stopper`, `fallback_extractor`, `no_acute_calibration`.
- **Method:** Monkey-patches specific functions / policy fields to disable subsystems, runs the same personas, compares **BDI totals**, **mean absolute error vs baseline**, **average turns**, and **monotonicity checks** across persona ordering (useful if personas are ordered by severity in mock settings).
- **Report:** JSON under `newVersion15Mar/reports/component_ablation_report.json`.

*Note:* The checked-in sample report reflects a **quick mock / reduced-turn** configuration in places; treat numbers as **illustrative** unless you rerun a full standardized protocol for the paper.

**Symptom difficulty (`scripts/analyze_symptom_difficulty.py`):**

- Uses **TalkDep** transcripts: compares **fallback** (`extract_symptoms_fallback`) vs **LLM extractor** per symptom.
- Surfaces **disagreement rates** and “severe miss” patterns — useful for an **error analysis** subsection.
- Example finding in `newVersion15Mar/reports/symptom_difficulty_report.json`: symptoms like **Self-Criticalness**, **Loss of Pleasure**, **Sadness**, **Tiredness or Fatigue** show **high extractor-mode disagreement** (paper should cite **fresh runs** for exact tables).

**Probe caps (`scripts/analyze_probe_caps.py`):**

- Loads saved **submission** interactions per persona/run and counts how often each **symptom/group/topic** was targeted vs policy caps; reports **cap violations** if routing inferred from questions exceeds configured maxima.

---

## 10. Design rationale (for Discussion section)

1. **Hybrid system:** LLMs for **language understanding and question generation**, explicit rules for **safety progression** and **interview completeness**, plus **template/lexicon** features for **transparent** risk scoring.
2. **Acute risk:** Clinical credibility requires **graded follow-up** rather than stopping at the first ambiguous phrase; hence **ladder coverage** interacts with **stopping** and **score calibration**.
3. **Extractor specialization:** A dedicated extractor with **explicit instructions** on suicide language and “happy to die” style contradictions mitigates **optimism bias** in general chat models.
4. **Three runs:** Reflects that **a single threshold** rarely optimizes both ends of the precision–recall tradeoff in conversational triage.

---

## 11. Reproducibility checklist

1. Python deps: `pip install -r requirements.txt`.
2. Set `.env`: `DEEPSEEK_API_KEY`, and `HF_TOKEN` for real personas.
3. Main runs: `python run.py --run all --output-dir outputs/submission/final` (see root `README.md`).
4. Ablations / analyses: commands in `newVersion15Mar/README.md`.

---

## 12. Limitations (suggested for paper)

- **Label access:** Official test **reference BDI** for eRisk personas may be **hidden**; reported metrics on TalkDep or mock personas are **proxies**.
- **API variability:** DeepSeek outputs can drift with **API updates**; exact scores may differ across dates unless logs are frozen.
- **Embedding optional:** Template and memory paths depend on **local embedding availability** and flags.
- **Safety:** System is a **research prototype** for **simulated** patients, not a clinical device.

---

## 13. Key file map

| Path | Role |
|------|------|
| `run.py` | Main CLI for competition runs |
| `src/orchestrator.py` | Turn loop wiring |
| `src/agents/{prober,extractor,stopper,scorer,risk_router,template_evidence,evidence_memory,interview_banks}.py` | Agent implementations |
| `src/persona_client.py` | Real vs mock patient |
| `src/bdi_mapper.py`, `src/topic_hierarchy.py` | BDI strings, validation, topic routing |
| `knowledge/*.yaml` | Policies, banks, templates |
| `newVersion15Mar/scripts/*.py` | Ablation / analysis experiments |
| `scripts/eval_*.py`, `scripts/run_talkdep_eval.py` | External evaluation helpers |

---

## 14. Suggested paper section mapping

- **Task / motivation:** §1–2  
- **Architecture / methods:** §3–6, §7 (official runs)  
- **Rich examples (clusters, questions, lexicon, templates):** §4A  
- **Implementation details:** §4–5, §13  
- **Experiments:** §9 (TalkDep, ablations, difficulty, caps) + **your** official leaderboard results when available  
- **Ethics / limitations:** §12  

---

*Generated from the repository state to support downstream paper drafting; update §9 with your final experimental numbers after locking protocols and rerunning analyses. For quoted questions and rules, prefer §4A and the cited source files so wording stays faithful to the implementation.*
