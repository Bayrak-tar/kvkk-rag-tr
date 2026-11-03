# Manual answer evaluation

Generated answers are not scored as correct by the retrieval metrics. For a generation run, prepare a review sheet from its CSV:

```powershell
python -m eval.run --config configs/eval-pilot-generation.json
python -m eval.prepare_answer_review results/preliminary/pilot-draft-generation-hybrid-qwen3.csv results/preliminary/pilot-draft-generation-review.csv
```

Have a reviewer compare the question, reference answer, retrieved article IDs, citations, excerpts, and generated answer. Fill the four score columns with integers from 1 to 5, set the abstention fields where applicable, and record the reviewer and date. The CSV deliberately starts with blank scores.

| Score | 1 | 3 | 5 |
| --- | --- | --- | --- |
| Correctness | Materially wrong | Partly correct, material issues | Correct against the cited official source |
| Completeness | Does not answer the main question | Main point present, important conditions omitted | Covers all answerable points and relevant conditions in the reference |
| Faithfulness | Material claims are unsupported or contradicted | Some support, but at least one claim is unclear | Each material claim is supported by retrieved text |
| Citation support | Citations do not support the answer | Some claims have matching sources, some do not | Each material claim has a source that directly supports it |

Use 2 and 4 for intermediate cases. Do not treat the model's `verification_status` as a human score; it is a separate automated signal. After annotations are complete, aggregate only filled ratings:

```powershell
python -m eval.summarize_answer_reviews results/preliminary/pilot-draft-generation-review.csv results/preliminary/pilot-draft-generation-review-summary.json
```

The OOD/in-domain answerability set is in `eval/questions/abstention_seed.jsonl`. All entries are `draft`; they must be reviewed before their metrics are presented as validated. Run it with the local configured model:

```powershell
python -m eval.run_abstention --config configs/eval-abstention.json
```

This writes per-question outputs and abstention precision/recall, false-refusal rate, unsafe OOD answer rate, and latency to `results/abstention/`. The runner labels unreviewed runs `preliminary_unreviewed` and records corpus/config/question hashes.
