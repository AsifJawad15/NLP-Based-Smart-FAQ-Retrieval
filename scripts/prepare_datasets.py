"""Download, convert, and validate the two Phase-1 FAQ corpora.

Examples:
    python scripts/prepare_datasets.py download
    python scripts/prepare_datasets.py convert
    python scripts/prepare_datasets.py validate
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datasets import load_dataset
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_faq_dataset, load_query_dataset  # noqa: E402


DATA_ROOT = PROJECT_ROOT / "data"
STAGING_DIR = DATA_ROOT / "staging"
TARGET_FAQS = 500
RANDOM_SEED = 42

SOURCES = {
    "university": {
        "dataset": "houcine-bdk/cpath-mcgill-ubc",
        "raw_file": "university_raw.csv",
    },
    "ecommerce": {
        "dataset": "NebulaByte/E-Commerce_FAQs",
        "raw_file": "ecommerce_raw.csv",
    },
}

OFFICIAL_UNIVERSITY_FAQ_SOURCES = [
    {
        "url": "https://www.oise.utoronto.ca/registrar-students/admissions/faq",
        "category": "admissions",
        "parser": "accordion",
    },
    {
        "url": "https://www.utoronto.ca/convocation/frequently-asked-questions",
        "category": "convocation",
        "parser": "accordion",
    },
    {
        "url": "https://www.utoronto.ca/alerts-faqs",
        "category": "campus_safety",
        "parser": "accordion",
    },
    {
        "url": "https://www.utoronto.ca/smoke-free/faqs",
        "category": "campus_policy",
        "parser": "accordion",
    },
    {
        "url": (
            "https://you.ubc.ca/jump-start-vancouver/"
            "jump-start-vancouvers-frequently-asked-questions/"
        ),
        "category": "student_orientation",
        "parser": "strong_siblings",
    },
    {
        "url": "https://www.utm.utoronto.ca/future-students/admissions/faq",
        "category": "admissions",
        "parser": "accordion",
    },
    {
        "url": "https://www.utm.utoronto.ca/future-students/transfer-credits/faq",
        "category": "transfer_credits",
        "parser": "accordion",
    },
    {
        "url": "https://www.sgs.utoronto.ca/future-students/faq/",
        "category": "graduate_admissions",
        "parser": "kadence",
    },
    {
        "url": (
            "https://www.utsc.utoronto.ca/utscinternational/"
            "immigration-frequently-asked-questions"
        ),
        "category": "international_students",
        "parser": "accordion",
    },
    {
        "url": (
            "https://undergrad.engineering.utoronto.ca/first-year-office-2/"
            "first-year-updates-deadlines/first-year-frequently-asked-questions/"
        ),
        "category": "first_year_support",
        "parser": "fl_accordion",
    },
    {
        "url": (
            "https://socialwork.utoronto.ca/practicum/for-students/"
            "frequently-asked-questions-by-students/"
        ),
        "category": "practicum",
        "parser": "arconix",
    },
]

GENERAL_UNANSWERABLE = [
    "Who won the latest international football tournament?",
    "What will the weather be tomorrow morning?",
    "Can you write a poem about the moon?",
    "How do I repair a leaking kitchen tap?",
    "What is the current price of Bitcoin?",
    "Which candidate won the national election?",
    "How long should I bake a chocolate cake?",
    "Can you diagnose the pain in my shoulder?",
    "What is the fastest route to the airport?",
    "Which movie should I watch tonight?",
    "How do I train a puppy to sit?",
    "What are today's cricket scores?",
    "Can you translate this paragraph into Japanese?",
    "How can I grow tomatoes on a balcony?",
    "What is the best camera for wildlife photography?",
    "How do I replace a car battery?",
    "What time does the next train leave?",
    "Can you calculate my income tax return?",
    "Which medicine should I take for a fever?",
    "How do I tune an acoustic guitar?",
    "What is the capital of every country in Europe?",
    "Can you predict next week's lottery numbers?",
    "How do I install solar panels on a roof?",
    "What ingredients are needed for vegetable soup?",
    "Where can I stream the championship match?",
]


def clean_spaces(value: object) -> str:
    """Convert a value to text and collapse repeated whitespace."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = re.sub(r"[\x00-\x1f\x7f]", " ", str(value))
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalized_question(text: str) -> str:
    """Normalize text for exact-like duplicate checks."""

    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def university_category(question: str) -> str:
    """Assign a clear course-project category using visible keyword rules."""

    text = question.lower()
    rules = [
        ("admissions", ["admission", "applicant", "apply", "application", "prerequisite", "qualifications"]),
        ("careers", ["career", "job", "employment", "graduates", "work after", "degree in"]),
        ("research", ["research", "thesis", "laboratory", "supervisor"]),
        ("student_services", ["resource", "service", "support", "counselling", "accessibility"]),
        ("tuition_financial_aid", ["tuition", "fee", "funding", "scholarship", "financial", "cost"]),
        ("courses_programs", ["course", "class", "curriculum", "subject", "program structure", "degree"]),
        ("campus_life", ["campus", "club", "student life", "residence", "housing"]),
        ("international_students", ["international", "visa", "language requirement", "english requirement"]),
        ("academic_rules", ["policy", "transfer", "credit", "enrol", "academic requirement"]),
    ]
    for category, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return category
    return "general_program_information"


def ecommerce_category(source_category: str, question: str) -> str:
    """Consolidate many source labels into ten presentation-friendly groups."""

    text = f"{source_category} {question}".lower()
    rules = [
        ("returns_refunds", ["cancel", "return", "refund", "replacement"]),
        ("shipping_delivery", ["shipping", "delivery", "courier", "pickup"]),
        ("payments", ["payment", "wallet", "phonepe", "credit card", "debit card", "emi"]),
        ("promotions_rewards", ["supercoin", "gift card", "discount", "coupon", "offer"]),
        ("accounts", ["login", "account", "password", "flipkart first", "plus"]),
        ("reviews", ["review", "rating"]),
        ("privacy_security", ["privacy", "security", "personal information", "fraud"]),
        ("warranties_installation", ["warranty", "installation", "service center"]),
        ("orders", ["order", "shopping", "cart", "checkout"]),
    ]
    for category, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return category
    return "products"


def quality_score(question: str, answer: str) -> float:
    """Give concise, informative rows a higher deterministic selection score."""

    question_words = len(question.split())
    answer_words = len(answer.split())
    score = 100.0
    score -= abs(question_words - 14) * 1.2
    score -= abs(answer_words - 55) * 0.25
    score -= max(0, answer_words - 110) * 0.8
    if question.endswith("?"):
        score += 3.0
    return score


def balanced_select(data: pd.DataFrame, count: int) -> pd.DataFrame:
    """Select high-quality rows while retaining every available category."""

    groups = {
        category: group.sort_values(
            ["quality_score", "question"], ascending=[False, True]
        ).to_dict("records")
        for category, group in data.groupby("category")
    }
    selected: list[dict[str, object]] = []
    category_names = sorted(groups)
    position = 0
    while len(selected) < count:
        added = False
        for category in category_names:
            if position < len(groups[category]):
                selected.append(groups[category][position])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        position += 1
    return pd.DataFrame(selected)


def _convert_university_legacy(raw: pd.DataFrame) -> pd.DataFrame:
    """Retained temporarily while the stricter converter is defined below."""

    records: list[dict[str, object]] = []
    allowed_question = re.compile(
        r"^(?:what are the admission requirements|what(?:'s| is) the admission process|"
        r"what qualifications do i need|how can i apply|what courses are offered|"
        r"what subjects will i study|what(?:'s| is) the course structure|"
        r"what career opportunities|what jobs can i get|where do .+ graduates|"
        r"what can i do with a degree|what are the research areas|"
        r"what research opportunities|what kind of research|"
        r"what resources are available for students)",
        re.IGNORECASE,
    )
    bad_question = re.compile(
        r"\b(report|archive|speaker series|news|newsletter|design day|summer letter|"
        r"offer of award|memorial award|travels from|student experience questions|"
        r"enhance your learning|what is mineral engineering)\b",
        re.IGNORECASE,
    )
    useful_source = re.compile(
        r"/(?:admission|admissions|academics|academic|program|programs|graduate|"
        r"undergraduate|student|students|research|course|courses|department|departments)/?",
        re.IGNORECASE,
    )
    bad_source = re.compile(
        r"/(?:article|articles|news|events?|archives?|reports?|letters?|awards?|people)/",
        re.IGNORECASE,
    )
    bad_answer_phrases = [
        "based on the program information, here's my advice",
        "remember to check the university's website regularly",
    ]

    for _, row in raw.iterrows():
        question = re.sub(
            r"^as cpath,\s*", "", clean_spaces(row.get("instruction")), flags=re.IGNORECASE
        )
        answer = re.sub(
            r"^as your canadian academic pathfinder,\s*(?:i(?:'|’)m happy to help\.\s*)?",
            "",
            clean_spaces(row.get("output")),
            flags=re.IGNORECASE,
        )
        source = clean_spaces(row.get("source_url"))
        q_words, a_words = len(question.split()), len(answer.split())

        if not question or not answer or not source.startswith("http"):
            continue
        if not 6 <= q_words <= 24 or not 12 <= a_words <= 125:
            continue
        if not allowed_question.search(question) or bad_question.search(question):
            continue
        if not useful_source.search(source) or bad_source.search(source):
            continue
        if any(phrase in answer.lower() for phrase in bad_answer_phrases):
            continue
        if re.search(r"\b(?:19|20)\d{2}\b", question):
            continue

        category = university_category(question)
        alignment_words = {
            "admissions": ["admission", "applicant", "apply", "required", "requirement", "eligible", "submit"],
            "careers": ["career", "work", "job", "industry", "professional", "graduate"],
            "research": ["research", "thesis", "faculty", "laboratory", "project"],
            "student_services": ["student", "support", "service", "resource"],
            "tuition_financial_aid": ["fee", "fund", "scholarship", "financial", "cost", "tuition"],
            "courses_programs": ["course", "program", "study", "curriculum", "class", "degree", "credit"],
        }
        if category in alignment_words and not any(
            word in answer.lower() for word in alignment_words[category]
        ):
            continue
        records.append(
            {
                "question": question,
                "answer": answer,
                "category": category,
                "source": source,
                "source_type": "public_dataset",
                "quality_score": quality_score(question, answer),
            }
        )

    data = pd.DataFrame(records)
    data["normalized"] = data["question"].map(normalized_question)
    data = data.drop_duplicates("normalized", keep="first").drop(columns="normalized")
    selected = balanced_select(data, TARGET_FAQS)
    if len(selected) < TARGET_FAQS:
        raise ValueError(f"Only {len(selected)} suitable University FAQs remained")
    selected.insert(0, "id", range(1, TARGET_FAQS + 1))
    return selected.drop(columns="quality_score")


def convert_ecommerce(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert and filter the source-backed e-commerce FAQ data."""

    excluded = re.compile(r"covid|flight|hotel|travel|insurance|loan", re.IGNORECASE)
    records: list[dict[str, object]] = []
    for _, row in raw.iterrows():
        source_category = clean_spaces(row.get("category"))
        question = clean_spaces(row.get("question"))
        answer = clean_spaces(row.get("answer"))
        source = clean_spaces(row.get("faq_url"))
        q_words, a_words = len(question.split()), len(answer.split())

        if excluded.search(f"{source_category} {question} {answer}"):
            continue
        if not question or not answer or not source.startswith("http"):
            continue
        if not 4 <= q_words <= 40 or not 5 <= a_words <= 150:
            continue

        category = ecommerce_category(source_category, question)
        records.append(
            {
                "question": question,
                "answer": answer,
                "category": category,
                "source": source,
                "source_type": "official_web_via_public_dataset",
                "quality_score": quality_score(question, answer),
            }
        )

    data = pd.DataFrame(records)
    data["normalized"] = data["question"].map(normalized_question)
    data = data.drop_duplicates("normalized", keep="first").drop(columns="normalized")
    selected = balanced_select(data, TARGET_FAQS)
    if len(selected) < TARGET_FAQS:
        raise ValueError(f"Only {len(selected)} suitable E-commerce FAQs remained")
    selected.insert(0, "id", range(1, TARGET_FAQS + 1))
    return selected.drop(columns="quality_score")


def university_question_category(question: str) -> str:
    """Classify only CPath prompt families that map to useful FAQ intents."""

    patterns = [
        (
            "admissions",
            r"^(?:what are the prerequisites|what are the admission requirements|"
            r"what(?:'s| is) the admission process|what qualifications do i need|"
            r"how can i apply)",
        ),
        (
            "careers",
            r"^(?:what are the career prospects|what career opportunities|"
            r"what jobs can i get|where do .+ graduates|"
            r"what can i do with a degree)",
        ),
        (
            "research",
            r"^(?:what are the research strengths|tell me about research|"
            r"what are the research areas|what research opportunities|"
            r"what kind of research)",
        ),
        (
            "courses_programs",
            r"^(?:what courses are offered|what subjects will i study|"
            r"what(?:'s| is) the course structure|tell me about the classes|"
            r"describe .+ curriculum)",
        ),
        (
            "program_overview",
            r"^(?:tell me about .+ program|can you explain the structure|"
            r"what are the key features|what makes .+ program .+ unique)",
        ),
    ]
    for category, pattern in patterns:
        if re.search(pattern, question, flags=re.IGNORECASE):
            return category
    return ""


def content_tokens(text: str) -> set[str]:
    """Return meaningful tokens for transparent question/source alignment."""

    project_stopwords = {
        "academic", "admission", "admissions", "after", "apply", "areas",
        "available", "canadian", "career", "completing", "course", "courses",
        "degree", "done", "features", "get", "graduate", "happy", "help",
        "jobs", "key", "kind", "masters", "mcgill", "need", "offered",
        "opportunities", "pathfinder", "phd", "process", "program", "programs",
        "qualifications", "requirements", "research", "resources", "structure",
        "student", "students", "study", "subjects", "tell", "typically", "ubc",
        "undergraduate", "unique", "university", "uoft", "work",
    }
    words = set(re.findall(r"[a-z]{3,}", text.lower()))
    return words - set(ENGLISH_STOP_WORDS) - project_stopwords


def convert_university(
    raw: pd.DataFrame, official_faqs: pd.DataFrame
) -> pd.DataFrame:
    """Convert strong CPath rows and fill the remainder with official FAQs."""

    bad_question = re.compile(
        r"\b(?:report|archive|speaker series|news|newsletter|design day|"
        r"summer letter|offer of award|memorial award|travels from|"
        r"abortion|aborto|table of cases|projects? 20\d\d|see what our alumni|"
        r"supporting documents?|contact us|faqs?|frequently asked|"
        r"upper-year applicants|professional training opportunities|"
        r"graduate research days|community outreach activities|"
        r"external financial support|fellowships?|scholarships?|"
        r"fees and financing|forms? (?:and|&) policies|events?|"
        r"strategic plan|welcome|home page|annual report|faculty profiles?|"
        r"employment, accommodation and more|seminar|workshop|conference|"
        r"symposium|lecture|appointed|elected|wins?|receives?|homeward bound|"
        r"retrospective|student presentations?|roadmap|practicum stories|"
        r"student section|student experience questions|about the school|"
        r"accreditation|great city|publications|departmental awards|"
        r"external awards|"
        r"buildings? & facilities|booking a lobby|industry)\b",
        re.IGNORECASE,
    )
    bad_source = re.compile(
        r"(?:/channels/event/|/engineering/article/|/article/|impact-report|"
        r"retrospective|/timetable/|academic_year=|graduate-research-days|"
        r"forms-policies|scholarships-funding|program-brochures|"
        r"20(?:0\d|1\d|2\d)|/faq/?$|intern-reports?|summer-letters?|"
        r"thesis-abstracts?|abortion-law|/people/|listserve|commentar|"
        r"publications|manuals-advocacy|buildings-facilities|celebration-ie)",
        re.IGNORECASE,
    )
    bad_answer_phrases = [
        "based on the program information, here's my advice",
        "based on the program information, here’s my advice",
        "remember to check the university",
        "graduates of this program have excellent career prospects",
        "the program offers research opportunities in these exciting areas",
        "admissions are suspended",
        "this class is now concluded",
    ]
    alignment_words = {
        "admissions": {
            "admission", "applicant", "apply", "application", "requirement",
            "required", "eligible", "submit", "deadline", "gpa", "transcript",
            "documents", "portfolio",
        },
        "careers": {
            "career", "work", "job", "employment", "industry", "professional",
            "graduate", "employer", "opportunities", "practice",
        },
        "research": {
            "research", "thesis", "faculty", "laboratory", "project",
            "investigate", "study", "scholarship",
        },
        "courses_programs": {
            "course", "program", "study", "curriculum", "class", "degree",
            "credit", "year", "term", "semester", "requirements",
        },
        "program_overview": {
            "program", "degree", "students", "study", "curriculum", "training",
            "education", "offers",
        },
    }

    records: list[dict[str, object]] = []
    for _, row in raw.iterrows():
        question = re.sub(
            r"^as cpath,\s*", "", clean_spaces(row.get("instruction")),
            flags=re.IGNORECASE,
        )
        answer = re.sub(
            r"^as your canadian academic pathfinder,\s*", "",
            clean_spaces(row.get("output")), flags=re.IGNORECASE,
        )
        answer = re.sub(
            r"^(?:i'm|i’m) happy to help\.\s*", "", answer,
            flags=re.IGNORECASE,
        )
        source = clean_spaces(row.get("source_url"))
        category = university_question_category(question)
        q_words, a_words = len(question.split()), len(answer.split())

        if not category or not source.startswith("http"):
            continue
        if not 6 <= q_words <= 24 or not 15 <= a_words <= 400:
            continue
        if bad_question.search(question) or bad_source.search(source):
            continue
        if any(phrase in answer.lower() for phrase in bad_answer_phrases):
            continue
        if re.search(r"\b20(?:0\d|1\d|2[0-5])\b", answer):
            continue

        answer_words = set(re.findall(r"[a-z]+", answer.lower()))
        alignment = len(answer_words & alignment_words[category])
        question_answer_overlap = len(content_tokens(question) & content_tokens(answer))
        question_source_overlap = len(
            content_tokens(question) & content_tokens(source.replace("-", " "))
        )
        if (
            alignment < 2
            or question_answer_overlap < 1
            or question_source_overlap < 1
        ):
            continue

        answer_length_penalty = abs(a_words - 100) / 100
        score = (
            alignment * 15
            + question_answer_overlap * 5
            + question_source_overlap * 2
            + quality_score(question, answer) / 20
            - answer_length_penalty
        )
        records.append(
            {
                "question": question,
                "answer": answer,
                "category": category,
                "source": source,
                "source_type": "public_dataset",
                "quality_score": score,
            }
        )

    data = pd.DataFrame(records)
    data["normalized_question"] = data["question"].map(normalized_question)
    data["normalized_answer"] = data["answer"].map(normalized_question)
    data = (
        data.sort_values(["quality_score", "question"], ascending=[False, True])
        .drop_duplicates("normalized_answer", keep="first")
        .drop_duplicates("normalized_question", keep="first")
        .drop(columns=["normalized_question", "normalized_answer"])
    )

    official_records: list[dict[str, object]] = []
    trusted_hosts = ("utoronto.ca", "ubc.ca", "mcgill.ca")
    for _, row in official_faqs.iterrows():
        question = clean_spaces(row.get("question"))
        answer = clean_spaces(row.get("answer"))
        source = clean_spaces(row.get("source"))
        category = clean_spaces(row.get("category"))
        if not source.lower().startswith("https://") or not any(
            host in source.lower() for host in trusted_hosts
        ):
            continue
        if question.lower().startswith("expand ") or "sub-menu" in question.lower():
            continue
        if not 3 <= len(question.split()) <= 45:
            continue
        if not 3 <= len(answer.split()) <= 500:
            continue
        official_records.append(
            {
                "question": question,
                "answer": answer,
                "category": category,
                "source": source,
                "source_type": "official_public_faq",
                "quality_score": 1_000.0,
            }
        )

    official = pd.DataFrame(official_records)
    official["normalized_question"] = official["question"].map(normalized_question)
    official = official.drop_duplicates("normalized_question", keep="first").drop(
        columns="normalized_question"
    )
    cpath_target = TARGET_FAQS - len(official)
    if cpath_target <= 0:
        raise ValueError("Official FAQ fallback unexpectedly exceeds the corpus target")
    if len(data) < cpath_target:
        raise ValueError(
            f"Only {len(data)} suitable CPath FAQs remained; need {cpath_target}"
        )

    selected = pd.concat(
        [data.head(cpath_target), official], ignore_index=True
    ).drop(columns="quality_score")
    selected.insert(0, "id", range(1, TARGET_FAQS + 1))
    return selected


def select_balanced_ids(data: pd.DataFrame, count: int, seed: int) -> list[int]:
    """Choose distinct FAQ ids with broad category coverage."""

    shuffled = data.sample(frac=1, random_state=seed)
    groups = {
        category: list(group["id"].astype(int))
        for category, group in shuffled.groupby("category")
    }
    selected: list[int] = []
    position = 0
    while len(selected) < count:
        for category in sorted(groups):
            if position < len(groups[category]):
                selected.append(groups[category][position])
                if len(selected) == count:
                    return selected
        position += 1
    raise ValueError(f"Could not select {count} balanced FAQ ids")


def paraphrase_question(question: str, variant: int) -> str:
    """Create a structurally different, deterministic evaluation query."""

    original = clean_spaces(question).rstrip("?").strip()
    lower = original.lower()
    patterns = [
        (r"^how (?:do|can) i (.+)$", [
            "What steps should I follow to {0}?",
            "Could you explain the process for {0}?",
            "I need help figuring out how to {0}.",
            "Please guide me through how to {0}.",
        ]),
        (r"^where can i (.+)$", [
            "Which page or place should I use to {0}?",
            "I need to {0}; where should I go?",
            "Where is the correct place for me to {0}?",
        ]),
        (r"^can i (.+)$", [
            "Is it possible for me to {0}?",
            "Am I allowed to {0}?",
            "I would like to know whether I may {0}.",
        ]),
        (r"^what are the admission requirements for (.+)$", [
            "What qualifications are needed to apply to {0}?",
            "Which requirements must an applicant meet for {0}?",
            "What do I need before applying to {0}?",
        ]),
        (r"^what(?:'s| is) the admission process for (.+)$", [
            "Which steps are involved in applying to {0}?",
            "Please explain how admission to {0} works.",
            "How would an applicant apply for {0}?",
        ]),
        (r"^what courses are offered in (.+)$", [
            "Which classes can students take in {0}?",
            "I want to know the course choices for {0}.",
            "What can a student study in {0}?",
        ]),
        (r"^what subjects will i study in (.+)$", [
            "Which subjects are included in {0}?",
            "Tell me about the curriculum for {0}.",
            "What classes make up {0}?",
        ]),
        (r"^what kind of research is done in (.+)$", [
            "Which research areas does {0} cover?",
            "What research work takes place in {0}?",
            "I want to know the research focus of {0}.",
        ]),
        (r"^what resources are available for students in (.+)$", [
            "Which student support resources does {0} provide?",
            "What help can students receive through {0}?",
            "Tell me about resources for students in {0}.",
        ]),
        (r"^what can i do with a degree in (.+)$", [
            "What career paths are available after studying {0}?",
            "Which opportunities follow a degree in {0}?",
            "Where could a qualification in {0} lead professionally?",
        ]),
        (r"^tell me about (.+)$", [
            "Could you provide information about {0}?",
            "I would like an overview of {0}.",
            "What should I know about {0}?",
        ]),
        (r"^why (.+)$", [
            "What is the reason that {0}?",
            "Could you explain why {0}?",
            "I want to understand why {0}.",
        ]),
        (r"^when (.+)$", [
            "At what time or stage {0}?",
            "Could you tell me when {0}?",
            "I need the timing for when {0}.",
        ]),
        (r"^what (?:is|are) (.+)$", [
            "Could you explain {0}?",
            "I need information about {0}.",
            "What should I know concerning {0}?",
        ]),
        (r"^will i (.+)$", [
            "I want to know whether I will {0}.",
            "Could you confirm if I will {0}?",
            "Please tell me whether I can expect to {0}.",
        ]),
        (r"^do i (.+)$", [
            "Is it necessary for me to {0}?",
            "Please confirm whether I need to {0}.",
            "Am I expected to {0}?",
        ]),
        (r"^is (.+)$", [
            "Please tell me whether {0}.",
            "Could you confirm if {0}?",
            "I would like to know whether {0}.",
        ]),
        (r"^are (.+)$", [
            "Please confirm whether {0}.",
            "I need to know if {0}.",
            "Could you tell me whether {0}?",
        ]),
        (r"^which (.+)$", [
            "I need to know which {0}.",
            "Could you identify which {0}?",
            "Please tell me which {0}.",
        ]),
        (r"^who (.+)$", [
            "Could you tell me who {0}?",
            "I need information about who {0}.",
            "Please identify who {0}.",
        ]),
    ]
    for pattern, templates in patterns:
        match = re.match(pattern, lower, flags=re.IGNORECASE)
        if match:
            topic = original[match.start(1) : match.end(1)]
            return templates[variant % len(templates)].format(topic)

    replacements = {
        " get ": " receive ",
        " buy ": " purchase ",
        " use ": " utilize ",
        " need ": " require ",
        " help ": " assistance ",
        " product ": " item ",
        " order ": " purchase ",
    }
    changed = f" {original.lower()} "
    for old, new in replacements.items():
        changed = changed.replace(old, new)
    changed = clean_spaces(changed)
    fallbacks = [
        "Could you explain this issue: {0}?",
        "I would like guidance about the following: {0}?",
        "What information is available regarding this: {0}?",
    ]
    return fallbacks[variant % len(fallbacks)].format(changed)


def build_query_sets(
    faq_data: pd.DataFrame,
    other_domain_data: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build validation and final test sets after FAQ ids are frozen."""

    answerable_ids = select_balanced_ids(faq_data, 180, seed)
    faq_by_id = faq_data.set_index("id")

    answerable_rows = []
    for position, faq_id in enumerate(answerable_ids):
        original = str(faq_by_id.loc[faq_id, "question"])
        query = paraphrase_question(original, position)
        if normalized_question(query) == normalized_question(original):
            raise ValueError(f"Generated query duplicates FAQ {faq_id}")
        answerable_rows.append(
            {"query": query, "expected_faq_id": faq_id, "is_answerable": True}
        )

    other_ids = select_balanced_ids(other_domain_data, 45, seed + 100)
    other_by_id = other_domain_data.set_index("id")
    cross_domain = [
        paraphrase_question(str(other_by_id.loc[faq_id, "question"]), i + 200)
        for i, faq_id in enumerate(other_ids)
    ]

    validation_unanswerable = cross_domain[:10] + GENERAL_UNANSWERABLE[:10]
    test_unanswerable = cross_domain[10:] + GENERAL_UNANSWERABLE[10:]
    if len(validation_unanswerable) != 20 or len(test_unanswerable) != 50:
        raise AssertionError("Unexpected unanswerable query count")

    validation_rows = answerable_rows[:30] + [
        {"query": query, "expected_faq_id": pd.NA, "is_answerable": False}
        for query in validation_unanswerable
    ]
    test_rows = answerable_rows[30:] + [
        {"query": query, "expected_faq_id": pd.NA, "is_answerable": False}
        for query in test_unanswerable
    ]

    validation = pd.DataFrame(validation_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    test = pd.DataFrame(test_rows).sample(frac=1, random_state=seed + 1).reset_index(drop=True)
    return validation, test


def flag_semantic_duplicates(data: pd.DataFrame, threshold: float = 0.88) -> pd.DataFrame:
    """Flag highly similar pairs for review without deleting them automatically."""

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(data["question"])
    similarities = cosine_similarity(matrix)
    flags = []
    for left in range(len(data)):
        for right in range(left + 1, len(data)):
            score = float(similarities[left, right])
            if score >= threshold:
                flags.append(
                    {
                        "left_id": int(data.iloc[left]["id"]),
                        "right_id": int(data.iloc[right]["id"]),
                        "similarity": score,
                        "left_question": data.iloc[left]["question"],
                        "right_question": data.iloc[right]["question"],
                    }
                )
    return pd.DataFrame(flags)


def extract_accordion_faqs(html: str) -> list[tuple[str, str]]:
    """Extract question/answer pairs linked by an accordion target id."""

    soup = BeautifulSoup(html, "html.parser")
    pairs: list[tuple[str, str]] = []
    for trigger in soup.find_all(["button", "a"]):
        question = clean_spaces(trigger.get_text(" ", strip=True))
        target = (
            trigger.get("data-target")
            or trigger.get("data-bs-target")
            or trigger.get("href", "")
        )
        if "?" not in question or not target.startswith("#"):
            continue
        answer_node = soup.select_one(target)
        if answer_node is None:
            continue
        answer = clean_spaces(answer_node.get_text(" ", strip=True))
        if answer.lower().startswith(question.lower()):
            answer = answer[len(question) :].strip()
        if answer:
            pairs.append((question, answer))
    return pairs


def extract_kadence_faqs(html: str) -> list[tuple[str, str]]:
    """Extract FAQ pairs from a Kadence accordion."""

    soup = BeautifulSoup(html, "html.parser")
    pairs: list[tuple[str, str]] = []
    for trigger in soup.select("button.kt-blocks-accordion-header"):
        question = clean_spaces(trigger.get_text(" ", strip=True))
        pane = trigger.find_parent(
            class_=lambda value: value and "kt-accordion-pane" in value.split()
        )
        answer_node = pane.select_one(".kt-accordion-panel-inner") if pane else None
        if "?" in question and answer_node is not None:
            pairs.append(
                (question, clean_spaces(answer_node.get_text(" ", strip=True)))
            )
    return pairs


def extract_fl_accordion_faqs(html: str) -> list[tuple[str, str]]:
    """Extract FAQ pairs from Beaver Builder accordion markup."""

    soup = BeautifulSoup(html, "html.parser")
    pairs: list[tuple[str, str]] = []
    for item in soup.select(".fl-accordion-item"):
        question_node = item.select_one(".fl-accordion-button-label")
        answer_node = item.select_one(".fl-accordion-content")
        if question_node is None or answer_node is None:
            continue
        question = re.sub(
            r"^q:\s*", "", clean_spaces(question_node.get_text(" ", strip=True)),
            flags=re.IGNORECASE,
        )
        answer = re.sub(
            r"^a:\s*", "", clean_spaces(answer_node.get_text(" ", strip=True)),
            flags=re.IGNORECASE,
        )
        if "?" in question and answer:
            pairs.append((question, answer))
    return pairs


def extract_arconix_faqs(html: str) -> list[tuple[str, str]]:
    """Extract FAQ pairs from Arconix FAQ containers."""

    soup = BeautifulSoup(html, "html.parser")
    pairs: list[tuple[str, str]] = []
    for item in soup.select(".arconix-faq-wrap"):
        question_node = item.select_one(".arconix-faq-title")
        answer_node = item.select_one(".arconix-faq-content")
        if question_node is None or answer_node is None:
            continue
        question = clean_spaces(question_node.get_text(" ", strip=True))
        answer = clean_spaces(answer_node.get_text(" ", strip=True))
        if "?" in question and answer:
            pairs.append((question, answer))
    return pairs


def extract_strong_sibling_faqs(html: str) -> list[tuple[str, str]]:
    """Extract FAQ pairs where a bold question precedes answer paragraphs."""

    soup = BeautifulSoup(html, "html.parser")
    pairs: list[tuple[str, str]] = []
    for strong in soup.find_all("strong"):
        question = clean_spaces(strong.get_text(" ", strip=True))
        if "?" not in question:
            continue
        parts: list[str] = []
        current = strong.parent.find_next_sibling()
        while current is not None and current.name not in {"h1", "h2", "h3", "h4"}:
            nested_strong = current.find("strong")
            if nested_strong is not None and "?" in nested_strong.get_text():
                break
            parts.append(clean_spaces(current.get_text(" ", strip=True)))
            current = current.find_next_sibling()
        answer = clean_spaces(" ".join(parts))
        if answer:
            pairs.append((question, answer))
    return pairs


def download_official_university_faqs() -> None:
    """Cache current official FAQ pages used only when CPath is insufficient."""

    rows: list[dict[str, str]] = []
    headers = {"User-Agent": "Smart-FAQ-academic-project/1.0"}
    for source in OFFICIAL_UNIVERSITY_FAQ_SOURCES:
        response = requests.get(source["url"], headers=headers, timeout=30)
        response.raise_for_status()
        parsers = {
            "accordion": extract_accordion_faqs,
            "arconix": extract_arconix_faqs,
            "fl_accordion": extract_fl_accordion_faqs,
            "kadence": extract_kadence_faqs,
            "strong_siblings": extract_strong_sibling_faqs,
        }
        pairs = parsers[source["parser"]](response.text)
        if not pairs:
            raise ValueError(f"No FAQ pairs found at {source['url']}")
        for question, answer in pairs:
            rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "category": source["category"],
                    "source": source["url"],
                }
            )
        print(f"Extracted {len(pairs)} official FAQs from {source['url']}")

    official = pd.DataFrame(rows)
    official["normalized"] = official["question"].map(normalized_question)
    official = official.drop_duplicates("normalized", keep="first").drop(
        columns="normalized"
    )
    output = STAGING_DIR / "university_official_faqs_raw.csv"
    official.to_csv(output, index=False, encoding="utf-8")
    print(f"Saved {len(official)} official FAQ rows to {output}")


def download_sources(domain: str) -> None:
    """Download selected Hugging Face sources into ignored staging CSVs."""

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    domains = SOURCES if domain == "all" else {domain: SOURCES[domain]}
    for name, source in domains.items():
        print(f"Downloading {name}: {source['dataset']}")
        dataset = load_dataset(source["dataset"], split="train")
        output = STAGING_DIR / source["raw_file"]
        dataset.to_pandas().to_csv(output, index=False, encoding="utf-8")
        print(f"Saved {len(dataset):,} raw rows to {output}")
    if domain in {"all", "university"}:
        download_official_university_faqs()


def convert_sources(domain: str) -> None:
    """Create final FAQ and query CSVs from downloaded source data."""

    requested = list(SOURCES) if domain == "all" else [domain]
    final_data: dict[str, pd.DataFrame] = {}
    for name in SOURCES:
        raw_path = STAGING_DIR / SOURCES[name]["raw_file"]
        if not raw_path.is_file():
            raise FileNotFoundError(f"Run download first; missing {raw_path}")
        raw = pd.read_csv(raw_path)
        if name == "university":
            official_path = STAGING_DIR / "university_official_faqs_raw.csv"
            if not official_path.is_file():
                raise FileNotFoundError(f"Run download first; missing {official_path}")
            official_faqs = pd.read_csv(official_path)
            final_data[name] = convert_university(raw, official_faqs)
        else:
            final_data[name] = convert_ecommerce(raw)

    for name in requested:
        directory = DATA_ROOT / name
        directory.mkdir(parents=True, exist_ok=True)
        faq_data = final_data[name]
        other_name = "ecommerce" if name == "university" else "university"
        validation, test = build_query_sets(
            faq_data, final_data[other_name], RANDOM_SEED + (0 if name == "university" else 10)
        )

        faq_data.to_csv(directory / "faq_dataset.csv", index=False, encoding="utf-8")
        validation.to_csv(directory / "validation_queries.csv", index=False, encoding="utf-8")
        test.to_csv(directory / "test_queries.csv", index=False, encoding="utf-8")

        flags = flag_semantic_duplicates(faq_data)
        flags.to_csv(
            STAGING_DIR / f"{name}_semantic_duplicate_flags.csv",
            index=False,
            encoding="utf-8",
        )
        print(
            f"{name}: {len(faq_data)} FAQs, {len(validation)} validation queries, "
            f"{len(test)} test queries, {len(flags)} duplicate-review flags"
        )


def validate_final(domain: str) -> None:
    """Validate finalized local data without using the network."""

    domains = list(SOURCES) if domain == "all" else [domain]
    for name in domains:
        directory = DATA_ROOT / name
        faq_data = load_faq_dataset(directory)
        faq_ids = set(faq_data["id"])
        validation = load_query_dataset(directory / "validation_queries.csv", faq_ids)
        test = load_query_dataset(directory / "test_queries.csv", faq_ids)

        if len(faq_data) != 500:
            raise ValueError(f"{name} must contain 500 FAQs, found {len(faq_data)}")
        if (len(validation), int(validation["is_answerable"].sum())) != (50, 30):
            raise ValueError(f"{name} validation split must be 30 answerable + 20 unanswerable")
        if (len(test), int(test["is_answerable"].sum())) != (200, 150):
            raise ValueError(f"{name} test split must be 150 answerable + 50 unanswerable")

        source_questions = faq_data.set_index("id")["question"]
        for _, row in pd.concat([validation, test]).query("is_answerable").iterrows():
            original = source_questions.loc[int(row["expected_faq_id"])]
            if normalized_question(row["query"]) == normalized_question(original):
                raise ValueError(f"Query duplicates its source FAQ: {row['query']}")

        print(
            f"VALID {name}: FAQs={len(faq_data)}, validation={len(validation)}, "
            f"test={len(test)}, categories={faq_data['category'].nunique()}"
        )


def parse_args() -> argparse.Namespace:
    """Parse the requested preparation mode and domain."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["download", "convert", "validate"])
    parser.add_argument(
        "--domain", choices=["all", "university", "ecommerce"], default="all"
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "download":
        download_sources(arguments.domain)
    elif arguments.mode == "convert":
        convert_sources(arguments.domain)
    else:
        validate_final(arguments.domain)
