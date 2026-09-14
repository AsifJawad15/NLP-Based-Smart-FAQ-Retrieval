# Smart FAQ Demo Rehearsal

Start the interface from the project directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address localhost
```

## 1. Easy KUET match

Enter:

```text
How many books may a KUET undergraduate borrow?
```

Expected behavior: FAQ 50 is accepted. The stored answer says an undergraduate
may borrow three books for one month and links to the KUET Central Library page.

## 2. Natural KUET paraphrase

Enter:

```text
I am an undergraduate. How many library books can I take at once?
```

Expected behavior: FAQ 50 is accepted. Open **Show NLP Details** and point out
the processed query, similarity, 0.47 threshold, FAQ ID, and top three matches.

## 3. Rejection

Enter:

```text
How can I mine cryptocurrency at home?
```

Expected behavior: the page says **No sufficiently relevant FAQ found** and
shows no answer. NLP Details may show nearest candidates, each marked unaccepted.

## 4. Seven-model research comparison

Open **Model Comparison**, select **University FAQ**, and enter:

```text
How do I request a transcript?
```

Expected behavior: every available research model completes independently in
one table. Point out its predicted FAQ, score, own threshold and decision. The
RNN and BiLSTM labels say experimental. Use the saved metrics expander to compare
benchmark outcomes, while explaining that raw similarity values from different
vector spaces are not directly comparable.

These are rehearsal checks. Questions asked later by the evaluator are ordinary
development feedback and are not labeled as a formal human benchmark.
