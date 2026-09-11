# Optional human benchmark for dataset annotators

To simply type questions and see an answer or None, run
`python evaluate.py manual --corpus university`. No CSV or FAQ IDs are needed.

This directory is for scientific scoring using independently established correct
labels. Annotators must check the corpus to assign these labels; ordinary testers
do not need to do this. The former `manual` CSV command is now `human-benchmark`.

Keep this header when adding questions to either CSV:

```csv
query,expected_faq_id,is_answerable
```

These files are separate from the synthetic validation/test sets and are not
discovered as FAQ corpora. Small trial sets do not establish overall performance.

## Collecting the questions

1. Keep the FAQ corpus and the validation-selected configurations frozen.
2. One team member prepares short intent descriptions from the FAQs. Another
   writes natural queries from those descriptions without seeing or copying the
   stored question wording. Do not use an AI generator for this evaluation set.
3. Aim for **20 answerable + 10 unanswerable questions per domain**. Use several
   categories, vocabulary changes, and natural sentence structures. Include
   plausible questions whose answers are absent from the chosen corpus, as well
   as clearly unrelated questions.
4. Independently check the complete query, stored answer, and intended FAQ id.
   An answerable query must have a valid id from its own corpus. An unanswerable
   query must have an empty id and no adequate answer anywhere in that corpus.
   Resolve ambiguous labels before scoring. Do not copy validation/test queries.
5. Record contributors and collection date below. Evaluate once the set is
   complete; report observed results without changing settings to improve them.

Contributors: pending. Collection date: pending.

## Format examples only - do not copy into measured files

Suppose a separate toy corpus has FAQ 42 about recovering a portal password:

```csv
query,expected_faq_id,is_answerable
"I cannot remember my portal password. How can I regain access?",42,True
"How do I teach my puppy to sit?",,False
```

The toy id is illustrative, not a verified label for either actual corpus.
CSV fields containing commas must be quoted. Use `True` or `False` labels.

## Running

From the Smart_FAQ project directory:

```powershell
python evaluate.py human-benchmark
python evaluate.py human-benchmark --corpus university
python evaluate.py human-benchmark --corpus ecommerce
python evaluate.py human-benchmark --corpus university --model phase3
```

Header-only files print **human evaluation pending** and produce no current
scores or reports. A populated file is checked with the normal query validator
and evaluated using its corpus's frozen configuration. Smaller sets can be used
while collecting data; the report includes the actual sample sizes.

TF-IDF outputs retain `reports/manual_<corpus>_evaluation.json` and
`reports/manual_evaluation_report.md`. Model groups write `manual_*` reports,
including `manual_comparison_report.md`, under the corresponding phase directory
(for example, `reports/phase3/`). Synthetic benchmark reports and corpus
configurations are not overwritten. The command cannot establish authorship;
only claim human-written results for questions your team actually collected.
