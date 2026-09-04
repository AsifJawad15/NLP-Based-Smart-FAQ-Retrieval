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


# The scraped pages were decoded twice, so a few punctuation marks arrive
# damaged. These are the only sequences observed in the raw downloads.
MOJIBAKE_REPAIRS = {
    "â€“": "-",
    "â€”": "-",
    "â€™": "'",
    "â€œ": '"',
    "â€": '"',
    "Â ": " ",
}

# U+FFFD is ambiguous: the same replacement character stands for an apostrophe
# in "program?s" and for a dash in "16 months ? 4 sessions", so the surrounding
# characters decide which one to restore.
REPLACEMENT = "�"
REPLACEMENT_REPAIRS = [
    (re.compile(REPLACEMENT + r"(?=s\b)"), "'"),
    (re.compile(REPLACEMENT + r"(?=(?:t|re|ve|ll|d|m)\b)"), "'"),
    (re.compile(r"\s*" + REPLACEMENT + r"\s*"), " - "),
    (re.compile(REPLACEMENT), ""),
]


# Constant assistant scaffolding produced by the CPath dataset. These strings
# are fixed, never paraphrased, so removing them literally is safe.
ASSISTANT_BOILERPLATE = [
    "I encourage you to explore the program details further and reach out to "
    "the university's admissions office if you have specific questions.",
    "I recommend reviewing these requirements carefully and ensuring you meet "
    "all prerequisites before applying.",
    "These courses are designed to provide you with comprehensive knowledge "
    "and skills in your field of study.",
    "Here are the admission requirements for this program:",
    "The program curriculum includes the following courses:",
]

# "This information comes directly from UOFT's official website." and friends.
PROVENANCE_PATTERN = re.compile(
    r"\s*This information comes directly from [A-Z]+'?s? official website\.\s*",
    re.IGNORECASE,
)

# Page headings that the source dataset wrongly used as a programme name.
HEADING_SLOT_PATTERN = re.compile(
    r"\b(?:program\s+details|program\s+information|programs?\s+overview|"
    r"admissions?\s+requirements?|degree\s+requirements|course\s+descriptions?|"
    r"course\s+details|testimonials?|prospective\s+students|final\s+year\s+average|"
    r"transcripts?|funding|prerequisites?|applicants?|office\s+of|"
    r"thesis\s+topic|after\s+the\s+defence|admissions|alumni|"
    r"student\s+services|how\s+to\s+apply|fees?\s+and\s+financing|"
    r"requirements?|advising|overview|information)\b",
    re.IGNORECASE,
)

# Question templates whose bracketed slot should name a real programme.
SLOT_PATTERNS = [
    r"^what can i do with a degree in (.+?) from \w+\?*$",
    r"^what are the admission requirements for (.+?) at \w+\?*$",
    r"^what(?:'s| is) the admission process for (.+?) at \w+\?*$",
    r"^what are the prerequisites for (.+?) at \w+\?*$",
    r"^how can i apply to (.+?) at \w+\?*$",
    r"^what subjects will i study in (.+?) at \w+\?*$",
    r"^what courses are offered in (.+?) at \w+\?*$",
    r"^what kind of research is done in (.+?) at \w+\?*$",
    r"^what are the research (?:areas|strengths) (?:in|of) (.+?) at \w+\?*$",
    r"^what are the career prospects for (.+?) graduates from \w+\?*$",
    r"^where do (.+?) graduates from \w+ typically work\?*$",
    r"^what career opportunities .*? (?:in|for) (.+?) at \w+\?*$",
    r"^tell me about (?:the )?(.+?) program at \w+\?*$",
    r"^can you explain the structure of (?:the )?(.+?) at \w+\?*$",
    r"^what jobs can i get after (?:studying |completing )?(?:the )?(.+?) at \w+\?*$",
    r"^tell me about the classes in (?:the )?(.+?) at \w+\?*$",
    r"^describe the (.+?) (?:curriculum|program) at \w+\.?\?*$",
    r"^describe the (.+?) at \w+\.?\?*$",
    r"^what makes the (.+?) (?:program |degree )?(?:at \w+ )?unique\?*$",
    r"^what research opportunities exist in (?:the )?(.+?) at \w+\?*$",
    r"^what are the key features of (?:the )?(.+?) at \w+\?*$",
]


def repair_text(text: str) -> str:
    """Repair the double-decoded punctuation seen in the scraped sources."""

    for damaged, fixed in MOJIBAKE_REPAIRS.items():
        text = text.replace(damaged, fixed)
    for pattern, fixed in REPLACEMENT_REPAIRS:
        text = pattern.sub(fixed, text)
    return text


def strip_assistant_boilerplate(answer: str) -> str:
    """Remove the fixed LLM-assistant scaffolding the project plan rejects."""

    cleaned = PROVENANCE_PATTERN.sub(" ", answer)
    for phrase in ASSISTANT_BOILERPLATE:
        cleaned = cleaned.replace(phrase, " ")
    return clean_spaces(cleaned)


def question_slot(question: str) -> str | None:
    """Return the programme name a CPath question template was filled with."""

    for pattern in SLOT_PATTERNS:
        match = re.match(pattern, question.strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


# Markers that begin a scraped page footer or staff-contact block. Everything
# from the earliest marker onwards is page furniture rather than an answer.
FURNITURE_MARKERS = [
    re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.IGNORECASE),
    re.compile(r"\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b"),  # Canadian postal code
    re.compile(r"©"),
    re.compile(r"\ball rights reserved\b", re.IGNORECASE),
    re.compile(r"\bManages [A-Z]", re.IGNORECASE),
    re.compile(r"\b\d{1,4} College Street\b", re.IGNORECASE),
]


def trim_page_furniture(answer: str) -> str:
    """Cut a scraped footer or contact block off the end of an answer.

    Input: one converted answer that may end in page furniture.
    Output: the answer text up to the earliest furniture marker.
    """

    cut = len(answer)
    for marker in FURNITURE_MARKERS:
        match = marker.search(answer)
        if match:
            cut = min(cut, match.start())
    trimmed = answer[:cut]

    # Drop a trailing partial sentence left behind by the cut.
    if cut < len(answer) and "." in trimmed:
        trimmed = trimmed[: trimmed.rfind(".") + 1]
    return clean_spaces(trimmed)


def is_structural_scrape(answer: str) -> bool:
    """Detect navigation menus and link tables that cannot be trimmed."""

    if answer.count(" - ") >= 5:
        return True
    if len(answer.split()) > 60 and answer.count(".") <= 1:
        return True
    if answer.lower().count("course details") >= 3:
        return True
    return False


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
        source_category = repair_text(clean_spaces(row.get("category")))
        question = repair_text(clean_spaces(row.get("question")))
        answer = repair_text(clean_spaces(row.get("answer")))
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
            r"^as cpath,\s*", "", repair_text(clean_spaces(row.get("instruction"))),
            flags=re.IGNORECASE,
        )
        answer = re.sub(
            r"^as your canadian academic pathfinder,\s*", "",
            repair_text(clean_spaces(row.get("output"))), flags=re.IGNORECASE,
        )
        answer = re.sub(
            r"^(?:i'm|i’m) happy to help\.\s*", "", answer,
            flags=re.IGNORECASE,
        )
        # The plan requires the assistant framing to be removed, not just the
        # opening "As CPath" phrase.
        answer = trim_page_furniture(strip_assistant_boilerplate(answer))
        source = clean_spaces(row.get("source_url"))
        category = university_question_category(question)
        q_words, a_words = len(question.split()), len(answer.split())

        if not category or not source.startswith("http"):
            continue
        if not 6 <= q_words <= 24 or not 15 <= a_words <= 400:
            continue
        # Reject questions built from a page heading instead of a programme
        # name, which produced items such as "what subjects will I study in
        # Final Year Average at UOFT?".
        slot = question_slot(question)
        if slot is None or HEADING_SLOT_PATTERN.search(slot):
            continue
        # Reject navigation menus, link tables, and scraped page footers.
        if is_structural_scrape(answer):
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
            # One category word is enough now that heading, boilerplate, and
            # structural-scrape rejection carry the quality checks.
            alignment < 1
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
        question = repair_text(clean_spaces(row.get("question")))
        answer = repair_text(clean_spaces(row.get("answer")))
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


# Content-word substitutions. Rewording the topic itself is what makes a
# generated query a real paraphrase instead of the original question with a
# new opening phrase.
CONTENT_SYNONYMS = {
    # University vocabulary
    "admission": "entry",
    "admissions": "entry",
    "requirements": "criteria",
    "requirement": "criterion",
    "prerequisites": "entry conditions",
    "apply": "put in an application",
    "application": "submission",
    "courses": "classes",
    "course": "class",
    "subjects": "topics",
    "subject": "topic",
    "programme": "degree",
    "program": "degree",
    "curriculum": "syllabus",
    "structure": "organisation",
    "study": "learn",
    "studying": "learning",
    "career": "professional",
    "careers": "professions",
    "prospects": "outlook",
    "research": "scholarly work",
    "graduates": "former students",
    "graduate": "former student",
    "tuition": "fees",
    "scholarship": "funding award",
    "deadline": "cut-off date",
    "transcript": "academic record",
    "enrol": "register",
    "faculty": "teaching staff",
    "thesis": "dissertation",
    "campus": "university grounds",
    "international": "overseas",
    # E-commerce vocabulary
    "order": "purchase",
    "orders": "purchases",
    "delivery": "shipment",
    "shipping": "dispatch",
    "refund": "money back",
    "return": "send back",
    "returns": "sending items back",
    "cancel": "call off",
    "cancellation": "calling off",
    "payment": "transaction",
    "payments": "transactions",
    "wallet": "digital purse",
    "account": "profile",
    "password": "passcode",
    "review": "rating",
    "reviews": "ratings",
    "warranty": "guarantee",
    "installation": "set-up",
    "coupon": "voucher",
    "discount": "price reduction",
    "seller": "vendor",
    "product": "item",
    "products": "items",
    "replacement": "exchange",
    "track": "follow",
    "invoice": "bill",
    "purchase": "buy",
    # Shared verbs and nouns
    "receive": "get",
    "obtain": "get",
    "need": "require",
    "change": "modify",
    "check": "verify",
    "find": "locate",
    "help": "assistance",
    "information": "details",
    "available": "on offer",
    "offered": "provided",
    "pay": "settle the amount",
    "paying": "settling payment",
    "using": "with",
    "option": "choice",
    "options": "choices",
    "mode": "method",
    "cost": "charge",
    "gift": "present",
    "store": "shop",
    "buy": "order",
    "use": "make use of",
    "item": "product",
    "items": "products",
    "different": "distinct",
    "cash": "money",
    "rewards": "benefits",
    "price": "amount",
    "feature": "function",
    "pick": "collect",
    "address": "location",
    "brand": "make",
    "authorised": "approved",
    "dealer": "retailer",
    "bank": "financial institution",
    "charges": "costs",
    "process": "procedure",
    "steps": "stages",
    "issue": "problem",
    "contact": "reach",
    "update": "revise",
    "select": "choose",
    "add": "include",
    "remove": "delete",
    "verify": "confirm",
    "eligible": "qualified",
    "status": "current state",
    "number": "identifier",
    "details": "particulars",
}

WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]*")


def substitute_content_words(text: str) -> str:
    """Replace known content words with plain-English equivalents.

    Input: any fragment of a source FAQ question.
    Output: the same meaning expressed with different vocabulary.
    """

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        replacement = CONTENT_SYNONYMS.get(word.lower())
        if replacement is None:
            return word
        if word[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement

    return WORD_PATTERN.sub(replace, text)


# Stems are rewritten structurally; the captured topic is reworded separately.
PARAPHRASE_PATTERNS = [
    (r"^how (?:do|can) i (.+)$", [
        "What steps should someone follow to {0}?",
        "What is the correct procedure to {0}?",
        "I am trying to {0} - what is the process?",
    ]),
    (r"^where can i (.+)$", [
        "Which page should someone visit to {0}?",
        "In which place is it possible to {0}?",
        "I am looking for the right location to {0}.",
    ]),
    (r"^can i (.+)$", [
        "Is it permitted to {0}?",
        "Am I allowed to {0}?",
        "Would it be possible to {0}?",
    ]),
    (r"^what are the admission requirements for (.+?)(?: at \w+)?$", [
        "Which entry criteria must be met for {0}?",
        "What must an applicant satisfy before joining {0}?",
        "Which qualifications does {0} expect?",
    ]),
    (r"^what(?:'s| is) the admission process for (.+?)(?: at \w+)?$", [
        "Which stages make up entry to {0}?",
        "How does someone work through entry to {0}?",
        "Walk me through joining {0}.",
    ]),
    (r"^what are the prerequisites for (.+?)(?: at \w+)?$", [
        "Which background knowledge is expected before {0}?",
        "What must be completed ahead of {0}?",
        "Which earlier work does {0} assume?",
    ]),
    (r"^how can i apply to (.+?)(?: at \w+)?$", [
        "Which steps lead to a submission for {0}?",
        "How is an application prepared for {0}?",
        "What is involved in putting an application into {0}?",
    ]),
    (r"^what courses are offered in (.+?)(?: at \w+)?$", [
        "Which classes can be taken within {0}?",
        "What teaching is provided inside {0}?",
        "Which units are available under {0}?",
    ]),
    (r"^what subjects will i study in (.+?)(?: at \w+)?$", [
        "Which topics are covered inside {0}?",
        "What material does {0} teach?",
        "Which areas of learning belong to {0}?",
    ]),
    (r"^what kind of research is done in (.+?)(?: at \w+)?$", [
        "Which scholarly work happens within {0}?",
        "What investigations are carried out in {0}?",
        "Which study areas does {0} pursue?",
    ]),
    (r"^what can i do with a degree in (.+?)(?: from \w+)?$", [
        "Which professions follow on from {0}?",
        "Where does a qualification in {0} lead?",
        "What working life comes after {0}?",
    ]),
    (r"^what are the career prospects for (.+?) graduates(?: from \w+)?$", [
        "What working outlook do former {0} students have?",
        "Which jobs tend to follow {0}?",
        "How do people who finish {0} progress professionally?",
    ]),
    (r"^where do (.+?) graduates(?: from \w+)? typically work$", [
        "In which places do former {0} students find employment?",
        "Which employers take on people who finish {0}?",
        "What kind of workplace suits a {0} leaver?",
    ]),
    (r"^what jobs can i get after (.+?)(?: at \w+)?$", [
        "Which roles open up once {0} is finished?",
        "What employment follows {0}?",
        "Which positions suit someone who completed {0}?",
    ]),
    (r"^can you explain the structure of (.+?)(?: at \w+)?$", [
        "How is {0} organised?",
        "What shape does {0} take?",
        "Describe the way {0} is arranged.",
    ]),
    (r"^describe the (.+?)(?: at \w+)?$", [
        "Give an outline of {0}.",
        "What does {0} involve?",
        "Summarise {0} for me.",
    ]),
    (r"^tell me about (?:the )?(.+?)(?: at \w+)?$", [
        "What should someone know about {0}?",
        "Give me an overview of {0}.",
        "Explain {0} briefly.",
    ]),
    (r"^what makes (?:the )?(.+?) unique$", [
        "What sets {0} apart from similar options?",
        "Which features distinguish {0}?",
        "Why would someone pick {0} over the alternatives?",
    ]),
    (r"^what research opportunities exist in (.+?)(?: at \w+)?$", [
        "Which openings for scholarly work sit inside {0}?",
        "What investigative work can be joined within {0}?",
        "Where can study projects be found in {0}?",
    ]),
    (r"^why (.+)$", [
        "For what reason {0}?",
        "What lies behind the fact that {0}?",
        "Explain the cause: {0}.",
    ]),
    (r"^when (.+)$", [
        "At which point {0}?",
        "What is the timing for when {0}?",
        "Tell me the moment at which {0}.",
    ]),
    (r"^what (?:is|are) (.+)$", [
        "Explain {0}.",
        "Give me the details of {0}.",
        "What should be understood about {0}?",
    ]),
    (r"^will i (.+)$", [
        "Should I expect to {0}?",
        "Is it the case that I would {0}?",
        "Confirm whether someone would {0}.",
    ]),
    (r"^do i (.+)$", [
        "Is it necessary to {0}?",
        "Must someone {0}?",
        "Am I expected to {0}?",
    ]),
    (r"^(?:is|are) (.+)$", [
        "Confirm whether {0}.",
        "Tell me if {0}.",
        "I want to establish whether {0}.",
    ]),
    (r"^which (.+)$", [
        "Identify which {0}.",
        "Tell me which {0}.",
        "Point me to which {0}.",
    ]),
    (r"^who (.+)$", [
        "Name the person who {0}.",
        "Tell me who {0}.",
        "Identify who {0}.",
    ]),
    (r"^what (.+)$", [
        "Tell me what {0}.",
        "Clarify what {0}.",
        "I need to establish what {0}.",
    ]),
]

# Used only when no structural pattern matches. These still reword the
# content, so they never reproduce the source question verbatim.
STRUCTURAL_FALLBACKS = [
    "Please clarify the following point: {0}.",
    "I need guidance on this matter: {0}.",
    "Explain how this works: {0}.",
]


def break_verbatim_reuse(candidate: str, original: str, variant: int) -> str:
    """Reword a query that still repeats its source question word for word.

    Input: a generated query, its source question, and the variant number.
    Output: the query, restructured if it still contains the source verbatim.
    """

    if normalized_question(original) not in normalized_question(candidate):
        return candidate

    # Strip the opening interrogative so the sentence is rebuilt as a request
    # rather than repeated as a question.
    body = re.sub(
        r"^(?:what|which|when|where|why|who|how|can|could|do|does|did|is|are|"
        r"will|would|should|if)\b\s*", "", original, flags=re.IGNORECASE
    ).strip()
    body = re.sub(r"^(?:i|you|we|they)\b\s*", "", body, flags=re.IGNORECASE).strip()
    body = substitute_content_words(body).rstrip("?.! ")
    if len(body.split()) < 3:
        body = substitute_content_words(original).rstrip("?.! ")

    rebuilt = [
        f"Please explain {body}.",
        f"I need a clear description of {body}.",
        f"Describe for me {body}.",
    ][variant % 3]
    return clean_spaces(rebuilt)


def paraphrase_question(question: str, variant: int) -> str:
    """Create a structurally and lexically different evaluation query.

    Input: one frozen FAQ question and a deterministic variant number.
    Output: a reworded query that keeps the meaning but not the wording.
    """

    original = clean_spaces(question).rstrip("?.! ").strip()
    # The corpus appends the institution to most generated questions; keeping
    # it would leak an exact token into every paraphrase.
    original = re.sub(
        r"\s+(?:at|from)\s+(?:UOFT|UBC|MCGILL)\s*$", "", original, flags=re.IGNORECASE
    ).strip()
    lower = original.lower()

    for pattern, templates in PARAPHRASE_PATTERNS:
        match = re.match(pattern, lower, flags=re.IGNORECASE)
        if not match:
            continue
        topic = original[match.start(1) : match.end(1)].strip()
        reworded = substitute_content_words(topic)
        return break_verbatim_reuse(
            templates[variant % len(templates)].format(reworded), original, variant
        )

    reworded = substitute_content_words(original)
    reworded = reworded[0].lower() + reworded[1:] if reworded else reworded
    return break_verbatim_reuse(
        STRUCTURAL_FALLBACKS[variant % len(STRUCTURAL_FALLBACKS)].format(reworded),
        original,
        variant,
    )


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
