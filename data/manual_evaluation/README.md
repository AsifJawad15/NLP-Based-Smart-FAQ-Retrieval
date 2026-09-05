# Human evaluation: pending team input

The two CSVs intentionally contain only the shared header:

```csv
query,expected_faq_id,is_answerable
```

No human-written performance has been measured. These templates are not part of
the synthetic validation/test sets and are not discovered as FAQ corpora.

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
python evaluate.py manual
python evaluate.py manual --corpus university
python evaluate.py manual --corpus ecommerce
```

Header-only files print **human evaluation pending** and produce no current
scores or reports. A populated file is checked with the normal query validator
and evaluated using its corpus's frozen configuration. Smaller sets can be used
while collecting data; the report includes the actual sample sizes.

Outputs are `reports/manual_<corpus>_evaluation.json` and
`reports/manual_evaluation_report.md`. Synthetic benchmark reports and corpus
configurations are not overwritten. The command cannot establish authorship;
only claim human-written results for questions your team actually collected.
