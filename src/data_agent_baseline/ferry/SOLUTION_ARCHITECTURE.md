# Final Solution Architecture — KDD Cup 2026 Data Agent

Model: `qwen/qwen3.5-35b-a3b` (via OpenRouter). Generic, B-board-safe design — no
hardcoded answers, no dataset-specific literals.

## End-to-end flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ ENTRY                                                                          │
│   Docker:  main.py            Local:  scratch/full_benchmark.py                │
│   env: MODEL_API_URL / MODEL_API_KEY / MODEL_NAME                              │
└───────────────┬────────────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ PREPROCESS  preprocess.py                                                      │
│   /input  ──(additive, signature-gated; never mutates originals)──▶ input_rw   │
└───────────────┬────────────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ RUNNER  run/runner.py     ThreadPool (max_workers=4), per-task timeout         │
│   for each task ─────────────────────────────────────────────────────────────▶│
└───────────────┬────────────────────────────────────────────────────────────────┘
                │  one task
                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ ReActAgent  agents/react.py                  (loop: max_steps=30)              │
│                                                                                │
│   ┌────────────────────────────────────────────────────────────────────┐      │
│   │ 1. build_messages()  agents/prompt.py                              │      │
│   │      system prompt + rules + tool descriptions + task question      │      │
│   └───────────────┬────────────────────────────────────────────────────┘      │
│                   ▼                                                            │
│   ┌────────────────────────────────────────────────────────────────────┐      │
│   │ 2. model.complete()  agents/model.py  (OpenAIModelAdapter)         │      │
│   │      qwen3.5-35b-a3b  •  process-timeout + SIGALRM (TLS-hang safe)   │      │
│   │      retry on transient errors                                      │      │
│   └───────────────┬────────────────────────────────────────────────────┘      │
│                   ▼                                                            │
│   ┌────────────────────────────────────────────────────────────────────┐      │
│   │ 3. parse_model_step()  →  thought + actions[]  (parallel actions)   │      │
│   └───────────────┬────────────────────────────────────────────────────┘      │
│                   ▼                                                            │
│   ┌────────────────────────────────────────────────────────────────────┐      │
│   │ 4. tools.execute()  tools/registry.py   (ThreadPool, parallel)      │      │
│   │                                                                     │      │
│   │   READ/INSPECT                QUERY ENGINES            TERMINATE     │      │
│   │   ─────────────               ─────────────           ─────────     │      │
│   │   list_context                execute_context_sql      answer ──┐    │      │
│   │   read_csv / read_json        (sqlite/.db)                      │    │      │
│   │   read_doc                    execute_context_duckdb ◀NEW       │    │      │
│   │   inspect_sqlite_schema        (SQL over ALL csv+json,          │    │      │
│   │   inspect_context_tables ◀NEW  cross-file joins)                │    │      │
│   │                               execute_python (pandas, 30s)      │    │      │
│   └────────────────────────────────────────────────────────────────┼────┘      │
│                                                                      │          │
│                                              ┌───────────────────────▼───────┐  │
│                                              │ ANSWER GATE (registry._validate│  │
│                                              │   _answer)                     │  │
│                                              │                                │  │
│                                              │  validate_answer() ◀NEW        │  │
│                                              │   tools/answer_validator.py    │  │
│                                              │   GENERIC checks only:         │  │
│                                              │   • helper/proof cols leaked   │  │
│                                              │   • LIMIT 1 on min/max (ties)  │  │
│                                              │   • aggregate drops zeros      │  │
│                                              │                                │  │
│                                              │  issues? ──yes──▶ ok=False,    │  │
│                                              │     feedback → agent retries   │  │
│                                              │     (CAP: ≥2 prior attempts    │  │
│                                              │      → accept; no timeout loop)│  │
│                                              │  clean? ──▶ accept (terminal)  │  │
│                                              └───────────────┬────────────────┘  │
│                   ▲   loop until terminal or max_steps       │ AnswerTable       │
│                   └─────────────────────────────────────────┘                   │
└───────────────┬────────────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ OUTPUT   prediction.csv  ──▶  /output/<task_id>/prediction.csv                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

## What is NEW vs the baseline (this session)

| Component | File | Why it generalizes to B-board |
|---|---|---|
| `execute_context_duckdb` | `tools/duckdb.py` | Real SQL over the ~97% of files that are CSV/JSON; cross-file joins; avoids pandas/JSON-escaping failures. Auto-discovers whatever files exist → zero task coupling. |
| `inspect_context_tables` | `tools/duckdb.py` | Lists CSV/JSON as queryable tables so the agent learns names before querying. |
| `validate_answer` (answer gate) | `tools/answer_validator.py` | Self-critique on the 3 most common projection/aggregation mistakes. Keys only off question wording + table shape — no literals. |

## B-board safety properties

1. **No hardcodes.** No place names, ids, table/column names, or magic thresholds in any
   new code. (kdd_gohan's Layer-2 task-specific validator + auto-corrector were deliberately
   NOT adopted.)
2. **Validator cannot regress correctness.** It only *rejects* (returns `ok=False`); it never
   rewrites an answer. Worst case it costs steps, never converts a right answer to wrong.
3. **No timeout loop.** After 2 prior `answer` attempts the gate stops firing and accepts.
4. **High precision.** Each check fires only on a high-confidence pattern (verified against a
   true-positive + false-positive battery).

## Tuning knobs

- Agent: `max_steps=30`, `model_call_timeout_seconds=300`
- Runner: `max_workers=4`, `task_timeout_seconds=1800`
- Validator retry cap: `2` (in `registry._validate_answer`)
```
