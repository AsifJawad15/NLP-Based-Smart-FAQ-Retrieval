"""Exercise interactive CLI routing without changing user data or reports."""

import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
import evaluate
import main as terminal


class InteractiveTests(unittest.TestCase):
    def session(self, inputs=(), query=None, corpus="university"):
        fixture = Path(__file__).parent / "fixtures" / "university_sample.csv"
        config = {"display_name": "University Fixture", "remove_stopwords": False,
                  "lemmatize": False, "similarity_threshold": 0.95}
        output = io.StringIO()
        with contextlib.redirect_stdout(output), \
                patch.object(terminal, "discover_corpora", return_value={"university": fixture.parent}), \
                patch.object(terminal, "load_faq_dataset", return_value=pd.read_csv(fixture)), \
                patch.object(terminal, "load_corpus_config", return_value=config), \
                patch.object(terminal, "build_answerer", wraps=terminal.build_answerer) as build, \
                patch("builtins.input", side_effect=inputs):
            terminal.run_session(corpus, "tfidf", query)
        self.assertEqual(build.call_count, 1)
        return output.getvalue()

    def test_exact_match_and_rejection_in_one_session(self):
        faq = pd.read_csv(Path(__file__).parent / "fixtures" / "university_sample.csv")
        output = self.session([faq.iloc[0]["question"], "university crime", "exit"])
        self.assertIn(str(faq.iloc[0]["answer"]), output)
        self.assertIn("Similarity Score: 1.0000", output)
        self.assertIn("Acceptance threshold: 0.9500", output)
        self.assertIn("  3.", output)
        self.assertIn("Result: None", output)
        self.assertIn("(unaccepted)", output)

    def test_empty_input_menu_and_quit(self):
        output = self.session(["bad", "1", "", "quit"], corpus=None)
        self.assertIn("Available corpora:", output)
        self.assertIn("Please enter one of the displayed numbers", output)
        self.assertIn("Please enter a question", output)
        self.assertIn("Goodbye!", output)

    def test_one_shot_oov_and_eof(self):
        output = self.session(query="zzqqxx")
        self.assertIn("Result: None", output)
        self.assertIn("No candidates", output)
        self.assertNotIn("Goodbye!", output)
        self.assertIn("Goodbye!", self.session([EOFError()]))

    def test_unknown_corpus(self):
        with self.assertRaisesRegex(SystemExit, "Unknown corpus"):
            self.session(corpus="missing")

    def test_dispatch_and_defaults(self):
        with patch("sys.argv", ["evaluate.py", "manual"]), \
                patch.object(terminal, "run_session") as run, \
                patch.object(evaluate, "run_testing") as score:
            evaluate.main()
            run.assert_called_once_with(None, "tfidf", None)
            score.assert_not_called()
        with patch("sys.argv", ["evaluate.py", "human-benchmark", "--model", "phase3"]), \
                patch.object(evaluate, "run_testing") as score, \
                patch.object(evaluate, "run_tuning") as tune:
            evaluate.main()
            score.assert_called_once_with("all", "phase3", manual=True)
            tune.assert_not_called()

    def test_invalid_arguments(self):
        for args in [["manual", "--model", "phase3"], ["manual", "--corpus", "all"],
                     ["manual", "--query", " "], ["test", "--query", "hello"]]:
            with self.subTest(args=args), patch("sys.argv", ["evaluate.py", *args]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                evaluate.parse_args()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
