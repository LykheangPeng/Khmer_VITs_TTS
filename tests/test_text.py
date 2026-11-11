import pytest

from khmer_vits.text import Tokenizer, has_ambiguous_number, khmer_number, normalize_khmer


@pytest.mark.parametrize("text, expected", [
    ("២០២៥", "ពីរពាន់ម្ភៃប្រាំ"), ("15", "ដប់ប្រាំ"), ("100", "មួយរយ"), ("110", "មួយរយដប់"),
    ("21", "ម្ភៃមួយ"), ("10,000", "មួយម៉ឺន"), ("250000", "ពីរសែនប្រាំម៉ឺន"), ("1200000", "មួយលានពីរសែន"),
    ("0", "សូន្យ"), ("$5", "ប្រាំដុល្លារ"), ("50%", "ហាសិបភាគរយ"), ("១០០០៛", "មួយពាន់រៀល"),
])
def test_numbers_and_symbols(text, expected):
    assert normalize_khmer(text) == expected


def test_khmer_number_large():
    assert khmer_number(1_000_000_000) == "មួយពាន់លាន"


def test_cleanup_and_idempotence():
    s = "មុខម្ហូប​តាមដងផ្លូវ  «គឺ»  ស្ករ ២ ស្លាបព្រា។។"
    n = normalize_khmer(s)
    assert n == "មុខម្ហូបតាមដងផ្លូវ គឺ ស្ករ ពីរ ស្លាបព្រា។"
    assert normalize_khmer(n) == n


def test_sample_sentence():
    assert normalize_khmer("ប្រទេសកម្ពុជាមានប្រជាជន​ជាង ១៦ លាននាក់") == "ប្រទេសកម្ពុជាមានប្រជាជនជាង ដប់ប្រាំមួយ លាននាក់"


def test_ambiguous_numbers():
    assert has_ambiguous_number("ថ្លៃ 1.5 ដុល្លារ") and has_ambiguous_number("ម៉ោង ១២:៣០")
    assert not has_ambiguous_number("10,000 រៀល") and not has_ambiguous_number("ឆ្នាំ ២០២៥")


def test_tokenizer_blanks_and_unknown():
    tok = Tokenizer(["_", " ", "ក", "ា"])
    ids, unknown = tok.encode("កា x", add_blank=True, return_unknown=True)
    assert ids == [0, 2, 0, 3, 0, 1, 0] and unknown == ["x"]
    assert tok.encode("កា", add_blank=False) == [2, 3]
