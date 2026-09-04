# Corpus Sources and Manual Review Record

This file records where every FAQ came from, what was rejected during
preparation, and which review decisions were made by hand. It exists so the
corpora can be defended in a viva rather than taken on trust.

The running application never contacts Hugging Face or the web. It reads only
the finalized CSV files in `data/`. Downloading is a one-time preparation step.

## Summary

| | University FAQ | E-commerce FAQ |
| --- | --- | --- |
| Final FAQs | 500 | 500 |
| Categories | 14 | 10 |
| Distinct source URLs | 126 | 500 |
| Mean question length (words) | 12.4 | 12.9 |
| Mean answer length (words) | 98.9 | 38.0 |
| Validation queries | 50 (30 answerable + 20 unanswerable) | 50 (30 + 20) |
| Test queries | 200 (150 answerable + 50 unanswerable) | 200 (150 + 50) |

## University corpus

Two sources are combined because the primary dataset alone could not supply
500 records that survived review.

### Source 1 — CPath dataset (239 records, `source_type = public_dataset`)

[houcine-bdk/cpath-mcgill-ubc](https://huggingface.co/datasets/houcine-bdk/cpath-mcgill-ubc),
23,971 raw rows. It is an instruction-tuning dataset built by an assistant
persona ("CPath") over McGill, UBC, and University of Toronto programme pages,
so the raw text carries assistant framing that had to be removed.

Column mapping: `instruction → question`, `output → answer`,
`source_url → source`.

### Source 2 — Official university FAQ pages (261 records, `source_type = official_public_faq`)

The project plan requires that any shortfall be filled with source-backed FAQs
from official public FAQ pages, never with generated filler. Eleven public FAQ
pages on `utoronto.ca` and `ubc.ca` were scraped, covering admissions,
convocation, campus safety, campus policy, orientation, transfer credits,
graduate admissions, international students, first-year support, and practicum.
Each record keeps the exact page URL it came from.

### What the record-level review found and fixed

An audit of the previously frozen corpus found that 197 of 500 rows (39.4%)
carried a defect, all of them in the CPath half. Every class below was verified
by counting rows in the actual CSV before a fix was written.

| Defect | Rows affected | Resolution |
| --- | --- | --- |
| Assistant boilerplate left in the answer | 78 | Five constant strings and the "This information comes directly from X's official website" tag are stripped literally |
| Page heading used as the programme name | 85 (41 producing nonsensical questions) | Questions whose template slot matches heading vocabulary are rejected |
| Scraped page footer appended (staff emails, postal address, copyright) | 35 | The footer is trimmed and the prose above it is kept |
| Navigation menu or link table used as an answer | 6 | Rejected outright |
| Double-decoded punctuation | present | Repaired; `U+FFFD` is resolved by context, as it stands for an apostrophe in `program?s` but a dash in `16 months ? 4 sessions` |
| Typographic punctuation that a Windows console cannot print | present in both corpora | Folded to ASCII (curly quotes, en/em dashes, non-breaking hyphen and space, ellipsis, breadcrumb arrows). Before this, `python main.py` crashed with `UnicodeEncodeError` part-way through printing an answer |

Characters that are genuine content are deliberately **kept**: the rupee sign
`₹` in 13 e-commerce records, `É` and `ć` in university programme and personal
names, and a short Arabic fragment inside two architecture answers. `main.py`
sets `errors="replace"` on stdout so a limited console degrades these
gracefully instead of crashing.

Examples of questions that were removed, quoted from the previous corpus:

- *"what subjects will I study in Final Year Average at UOFT?"*
- *"what can I do with a degree in EMHI Program Details from UOFT?"*
- *"what are the admission requirements for Transcripts at UOFT?"*
- *"how can I apply to PhD: HPER Program Details Selecting a Suitable Thesis Topic After the Defence at UOFT?"*

After the fixes, the rebuilt corpus contains 0 boilerplate strings,
0 heading-derived programme slots, and 0 damaged characters.

One filter was deliberately relaxed. The topical-alignment gate previously
required two category words in the answer; it now requires one, because the
heading, boilerplate, and structural-scrape rejections added above are stricter
and more targeted than the blunt word count they replace. Without that change
only 210 clean CPath records remained, which is below the 239 needed.

## E-commerce corpus

[NebulaByte/E-Commerce_FAQs](https://huggingface.co/datasets/NebulaByte/E-Commerce_FAQs),
`source_type = official_web_via_public_dataset`. These are help-centre FAQs
scraped from a live retailer, so each of the 500 records keeps its own
`faq_url`.

Retained columns: `question`, `answer`, `category`, `faq_url → source`.
Excluded: blank rows, and any record matching travel, flight, hotel, insurance,
loan, or COVID material. The many source labels were consolidated into ten
presentation-friendly categories.

**Region-specific content is a known and accepted property of this corpus.**
The records mention rupee amounts, PhonePe, SuperCoins, Axis Bank, EMI plans,
and Indian delivery terms. This is preserved rather than edited because the
project plan requires source wording to be kept. The engine is generic; the
records are not.

Two categories hold a single record each (`privacy_security`, `reviews`). They
are kept because they are genuine records with valid sources, but they are too
small to support any per-category claim.

## Semantic duplicate review

High-similarity question pairs are flagged for manual review and are never
removed automatically. Exact and normalized duplicate questions are already
rejected by `validate_faq_data`, so only semantic pairs reach this stage.

All seven flagged pairs were adjudicated by reading both full records. **All
seven are DISTINCT; none were removed.**

| Corpus | Pair | Cosine | Decision |
| --- | --- | --- | --- |
| University | 158 / 209 | 1.000 | Distinct — Two Year vs One Year MMPA are different programmes with different entry rules |
| University | 288 / 293 | 1.000 | Distinct — *how* alerts arrive vs *how often* they arrive |
| E-commerce | 241 / 479 | 0.957 | Distinct — eligibility to apply vs what happens to the existing card |
| E-commerce | 289 / 295 / 301 | 0.885-0.889 | Distinct — same question asked of three different scooter brands, each with its own answer and URL |
| E-commerce | 469 / 475 | 1.000 | Distinct — *which* cards can be saved vs *how many* |

The flagging threshold is 0.88 on question text only. This is a documented
limitation: it does not detect near-identical *answers* attached to differently
worded questions, and lowering it surfaces more pairs that then need review.

## Evaluation query generation

Paraphrases are written only after the FAQ corpus is frozen, so a query can
never influence which records were selected.

The plan requires paraphrases to change sentence structure *and* wording. The
first implementation did not meet that bar: three generic fallback templates
wrapped the original question almost verbatim, so about 44% of answerable
queries contained their source question word for word, and mean source-token
coverage was 0.90. Top-1 accuracy of 1.000 was an artefact of that leakage.

The current generator substitutes content words, drops the institution suffix
that would otherwise leak an exact token into every query, restructures the
sentence, and rebuilds any candidate that still reproduces its source.

| | Before | After |
| --- | --- | --- |
| University mean source-token coverage | 0.900 | 0.587 |
| E-commerce mean source-token coverage | 0.916 | 0.766 |
| Queries containing their source verbatim | 44% | 0 |

E-commerce coverage stays higher because brand and product names
(Flipkart, Ather, BGauss, SuperCoins) are the discriminative content of the
question and must be kept; substituting them would make the query unanswerable.

Unanswerable queries are 45 cross-domain paraphrases drawn from the other
corpus plus 25 hand-written general-knowledge questions, split so that no query
appears in both the validation and test sets.

## Known limitations

- Generated paraphrases are not always fluent. Rebuilding a question that
  repeated its source can produce phrasing such as *"Please explain does
  'Preorder' or 'Forthcoming' mean."* The meaning is preserved and the wording
  differs, which is what the evaluation needs, but these are not human-written
  queries.
- The university corpus mixes an assistant-generated dataset with scraped
  official pages. The two halves differ in tone and answer length.
- Duplicate flagging inspects questions only, at a fixed 0.88 threshold.
- E-commerce content is region-specific, as described above.
