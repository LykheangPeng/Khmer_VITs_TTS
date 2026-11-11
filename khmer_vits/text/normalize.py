"""Khmer text normalisation: invisible characters, punctuation, numbers -> words."""
import re
import unicodedata


KHMER_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")


_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­឴឵"), None)


_PUNCT = str.maketrans({
    "“": None, "”": None, "„": None, '"': None, "«": None, "»": None,
    "‘": None, "’": None, "'": None, "`": None, "´": None,
    "(": " ", ")": " ", "[": " ", "]": " ", "{": " ", "}": " ", "<": " ", ">": " ",
    "•": " ", "*": " ", "#": " ", "_": " ", "/": " ", "\\": " ", "|": " ", "~": " ", "=": " ",
    "…": "។", "៚": "។", "៙": None, "៘": None, "。": "។",
    "–": "-", "—": "-", "‐": "-", "‑": "-", "−": "-",
    "？": "?", "！": "!", "，": ",", "、": ",", "：": ":", "；": ";",
    "\t": " ", "\n": " ", "\r": " ",
})


_UNITS = ["សូន្យ", "មួយ", "ពីរ", "បី", "បួន", "ប្រាំ", "ប្រាំមួយ", "ប្រាំពីរ", "ប្រាំបី", "ប្រាំបួន"]


_TENS = ["", "ដប់", "ម្ភៃ", "សាមសិប", "សែសិប", "ហាសិប", "ហុកសិប", "ចិតសិប", "ប៉ែតសិប", "កៅសិប"]


_BIG = [(100000, "សែន"), (10000, "ម៉ឺន"), (1000, "ពាន់"), (100, "រយ")]


_THOUSANDS_RE = re.compile(r"(?<![\d.,])\d{1,3}(?:[,.]\d{3})+(?![\d]|[.,]\d)")


_AMBIGUOUS_RE = re.compile(r"\d\s*[.,:/]\s*\d")


_NUM_RE = re.compile(r"\d+")


def khmer_number(n: int) -> str:
    """Integer -> Khmer words, e.g. 2025 -> ពីរពាន់ម្ភៃប្រាំ."""
    if n == 0:
        return _UNITS[0]
    out = []
    if n >= 1_000_000:
        out.append(khmer_number(n // 1_000_000) + "លាន")
        n %= 1_000_000
    for value, word in _BIG:
        q, n = divmod(n, value)
        if q:
            out.append(_UNITS[q] + word)
    q, r = divmod(n, 10)
    if q:
        out.append(_TENS[q])
    if r:
        out.append(_UNITS[r])
    return "".join(out)


def _expand_number(m):
    s = m.group(0)
    if (len(s) > 1 and s[0] == "0") or len(s) > 12:  # codes / phone numbers: digit by digit
        return "".join(_UNITS[int(c)] for c in s)
    return khmer_number(int(s))


def has_ambiguous_number(text: str) -> bool:
    """Decimals, times, fractions, dates... (anything but plain integers / 1,000-style groups)."""
    t = str(text).translate(KHMER_DIGITS)
    t = _THOUSANDS_RE.sub("0", t)
    return bool(_AMBIGUOUS_RE.search(t))


def normalize_khmer(text: str) -> str:
    t = unicodedata.normalize("NFC", str(text))
    t = t.translate(_INVISIBLE).translate(KHMER_DIGITS).translate(_PUNCT)
    t = re.sub(r"\$\s*(\d+(?:[,.]\d{3})*)", r"\1ដុល្លារ", t)
    t = t.replace("$", "ដុល្លារ").replace("%", "ភាគរយ").replace("៛", "រៀល").replace("&", " និង ")
    t = _THOUSANDS_RE.sub(lambda m: re.sub(r"[,.]", "", m.group(0)), t)
    t = _NUM_RE.sub(_expand_number, t)
    t = re.sub(r"([។៕?!,.:;\-])\1+", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
