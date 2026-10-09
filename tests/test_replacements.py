"""Turbo-Type-compatible word corrections, single-pass behavior and safe validation."""
import pytest

from app.core.replacements import ReplacementError, Replacements


@pytest.mark.parametrize("rules,text,expected", [
    ("wörf ; verve", "Wörf, WÖRF und wörf!", "Verve, VERVE und verve!"),
    ("wörf ; verve", "Wörfel und Abwörf, aber wörf!", "Wörfel und Abwörf, aber verve!"),
    ("Croc, Krog, Krok ; Groq ; 1", "Croc, krog und KROK", "Groq, Groq und Groq"),
    ("iphone ; iPhone ; 1", "Iphone IPHONE iphone", "iPhone iPhone iPhone"),
    ("iphone ; iPhone ; 0", "Iphone IPHONE iphone", "IPhone IPHONE iPhone"),
    ("chat gpt ; ChatGPT ; 1", "Chat\t GPT und CHAT\nGPT", "ChatGPT und ChatGPT"),
    ("a ; b\nb ; c", "a b", "b c"),
    ("new york ; NY\nnew york city ; New York City", "new york city", "New York City"),
    ("a, b ; X ; 1\nb, c ; Y ; 1", "a b c", "X Y Y"),
    ("foo ; bar ; 1", "_foo_ foo2 2foo éfoo fooé (foo)", "_bar_ foo2 2foo éfoo fooé (bar)"),
    ("c++ ; CPlus ; 1", "c++ and c++code", "CPlus and c++code"),
    ("input ; $1\\path ; 1", "input", "$1\\path"),
    ("i ; x ; 1", "İ ı I i", "x x x x"),
    ("", "untouched", "untouched"),
    ("a ; b", "", ""),
])
def test_replacements(rules, text, expected):
    assert Replacements(rules).apply(text)[0] == expected


def test_counts_and_source_lines_follow_later_duplicate_rules():
    engine = Replacements("a ; old\n\nb, a ; fixed ; 1\nc ; C ; 1")
    assert (engine.rule_count, engine.term_count) == (2, 3)
    assert engine.apply("a b a c") == ("fixed fixed fixed C", 4, [3, 4])
    assert engine.apply("no matches") == ("no matches", 0, [])


@pytest.mark.parametrize("broken", ["missing separator", "; target", "source ;", "a ; b ; 2", "a;b;1;extra"])
def test_invalid_rules_identify_line_without_exposing_contents(broken):
    with pytest.raises(ReplacementError) as error:
        Replacements("ok ; valid\n\n" + broken)
    assert error.value.line == 3
    assert broken not in str(error.value)
