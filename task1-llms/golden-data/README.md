# Ground Truth Data (Task 1 - LLM Personas)

This folder contains the reference labels used to evaluate participant predictions.

- `patients_data.jsonl`: one JSON object per persona/patient with `patient_id`, `patient_name`, `bdi_score`, and `patient_key_symptoms`.
- `bdi_symptoms_list.json`: canonical BDI-II symptom inventory (21 items).
- `symptom_mappings.json`: normalization map for symptom variants and synonyms used when matching predicted symptoms to canonical labels.
