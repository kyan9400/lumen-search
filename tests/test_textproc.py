import pytest

from lumen.textproc import (
    analyze,
    detect_language,
    highlight_spans,
    normalize,
    split_sentences,
    stem,
    term_spans,
)


def test_normalize_folds_case_yo_and_soft_hyphens() -> None:
    assert normalize("Ёлка") == "елка"
    assert normalize("pass\N{SOFT HYPHEN}word") == "password"
    assert normalize("ﬁle") == "file"  # NFKC expands ligatures


def test_analyze_drops_stop_words_and_stems_english() -> None:
    assert analyze("The passwords are stored in a database") == ["password", "store", "databas"]


def test_analyze_stems_russian_word_forms_to_one_term() -> None:
    forms = ["пароль", "пароля", "паролем", "паролей"]
    assert len({stem(normalize(f)) for f in forms}) == 1
    assert analyze("Хранение персональных данных") == ["хранен", "персональн", "дан"]


def test_analyze_splits_identifiers_on_punctuation() -> None:
    assert analyze("IA-5(1)") == ["ia", "5", "1"]
    assert analyze("NIST SP 800-63B") == ["nist", "sp", "800", "63b"]


def test_term_spans_point_into_the_original_text() -> None:
    text = "Strong Passwords, please."
    spans = term_spans(text)
    assert [(term, text[a:b]) for term, a, b in spans] == [
        ("strong", "Strong"),
        ("password", "Passwords"),
        ("pleas", "please"),
    ]


def test_highlight_spans_match_stems_not_surface_forms() -> None:
    text = "Each password is hashed; passwords are never logged."
    found = [text[a:b] for a, b in highlight_spans(text, set(analyze("password")))]
    assert found == ["password", "passwords"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Use Argon2id. It is memory-hard.", ["Use Argon2id.", "It is memory-hard."]),
        ("Hash it, e.g. with bcrypt. Then store it.", ["Hash it, e.g. with bcrypt.", "Then store it."]),
        ("See J. Smith for details. Done.", ["See J. Smith for details.", "Done."]),
        (
            "1. Verifiers SHALL require passwords. 2. They SHOULD allow paste.",
            ["1. Verifiers SHALL require passwords.", "2. They SHOULD allow paste."],
        ),
        ("Согласно ст. 5 закона. Иначе нельзя.", ["Согласно ст. 5 закона.", "Иначе нельзя."]),
        ("Это т.е. пример. Дальше!", ["Это т.е. пример.", "Дальше!"]),
        ("The minimum length is 8. Longer is better.", ["The minimum length is 8.", "Longer is better."]),
    ],
)
def test_split_sentences(text: str, expected: list[str]) -> None:
    assert split_sentences(text) == expected


def test_split_sentences_joins_wrapped_lines() -> None:
    assert split_sentences("A sentence that\nwraps lines.  Next one.") == [
        "A sentence that wraps lines.",
        "Next one.",
    ]


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("Passwords shall be at least fifteen characters long.", "en"),
        ("Хранение персональных данных должно осуществляться в форме", "ru"),
        ("Статья пять, принципы обработки, and article five principles", "mixed"),
        ("12345", "und"),
    ],
)
def test_detect_language(text: str, language: str) -> None:
    assert detect_language(text) == language
