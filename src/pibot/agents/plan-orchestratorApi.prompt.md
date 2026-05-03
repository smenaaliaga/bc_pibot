## Plan: API‑Based Classification + Macro Routing + Name Facts

Switch to HTTP endpoints for JointBERT/intent classification, reshape `ClassificationResult` to match API response fields, and route using `macro/context`. Store user names in `session_facts` with append semantics as a title‑case array, ignore birthdays, and return the gentle‑nudge message for non‑economic queries.

### Steps 4–6 steps, 5–20 words each
1. Add `PREDICT_URL` and `INTENT_CLASSIFIER_URL` to `config.py` and `.env` defaults.
2. Create a POST JSON helper (10s timeout) in `orchestrator/utils`.
3. Extend `ClassificationResult` in `orchestrator/classifier/classifier_agent.py` with API fields (`words`, `slot_tags`, `calc_mode`, `activity`, `region`, `investment`, `req_form`, `macro`, `context`).
4. Replace JointBERT call with `POST /predict`, normalize entities, then call intent‑classifier API to set `intent/context/macro`.
5. Update `orchestrator/routes/intent_router.py` to route by `macro/context`, detect name phrases, and store to `session_facts` under `user_name` (append, title case).
6. Emit the gentle‑nudge response when `macro=0` and no name fact was extracted.

### Further Considerations 1–3, 5–25 words each
1. Ensure name extraction handles “mi nombre es”, “yo soy”, “me llaman”, “me llamo”, “me dicen”.
2. Normalize user names to title case before appending.
3. Use array storage in `session_facts` for `user_name` values.
