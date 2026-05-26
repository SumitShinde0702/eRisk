# eRisk 2026 — Task 1 Instructions

## Conversational Depression Detection with LLM Personas (LoRA)

This task evaluates systems for conversational depression detection by interacting with LLM personas.
Participants must run conversations with each persona and submit their predictions together with the
conversation logs.

For detailed specifications of **required predictions**, **conversation constraints**, **file naming and formats**,
**rules**, and **reproducibility requirements**, see:

- https://erisk.irlab.org/Task1LLMs.html

---

## 1) LLM Personas and Model Setup

### Personas

- **20 LLM personas** will be released in total.
- Each persona was trained using **LoRA adapters** on top of the base model:
  - `meta-llama/Meta-Llama-3-8B-Instruct` (LLaMA 3 8B)

### Hugging Face release

- We will release **LoRA adapters only** (not full merged models).
- Each persona corresponds to **one adapter** (LoRA files).
- Personas will be published in the following Hugging Face collection:
  - https://huggingface.co/collections/irlab-udc/erisk2026

We will notify participants by email when new personas are released (with direct links).

### Code

We will provide starter conversational code for interacting with the personas.

Participants may use any implementation they prefer, as long as it complies with the official task
constraints described at:

- https://erisk.irlab.org/Task1LLMs.html

---

## 2) Timeline and Weekly Releases

### Release schedule

Every week, we will release **two personas**:

- **Release day:** Monday
- **Submission deadline for those personas:** Sunday (end of week)

### Calendar

- **First release window:** Monday, February 16, 2026 → Sunday, February 22, 2026 (Sunday AoE closes the window)
- **Last release window:** Monday, April 20, 2026 → Sunday, April 26, 2026 (Sunday AoE closes the window)

This schedule is aligned with the overall timeline so that results can be released by **May 12, 2026**.

---

## 3) Submitting Results

### Where to upload

For each persona, we will create a dedicated directory on the eRisk FTP server:

- **20 persona directories in total** (one per persona)

Each directory will accept uploads for:

- **Predictions** (your inferences for that persona)
- **Conversation logs** (full transcript / interaction trace)

### Deadline enforcement

At the end of each weekly window (Sunday), we will remove write permissions for the corresponding
persona directories to enforce the deadline.

### Number of runs

Teams may submit up to **3 runs / methods per persona**.

---

## 4) What to Submit (per persona)

For each persona, upload:

- **Predictions**, following the official schema and file format described at:
  - https://erisk.irlab.org/Task1LLMs.html
- **Conversation logs**, following the official logging requirements described at:
  - https://erisk.irlab.org/Task1LLMs.html

---

## 5) Official Guidelines (must-read)

All details about:

- required outputs (predictions)
- conversation constraints (turn/token limits, allowed strategies)
- file naming conventions and formats
- rules (automation/manual constraints, allowed tools)
- reproducibility requirements

are defined in the official instructions page:

- https://erisk.irlab.org/Task1LLMs.html