"""A signup-line grammar.

A legal line looks like this:

    name=maya grade=10 club=robotics

name= and club= take a lowercase word. grade= takes a whole number.
The three fields are in that order, with one space between them.
"""

import re

_LINE = re.compile(r"^name=([a-z]+) grade=([0-9]+) club=([a-z]+)$")
_LETTERS = set("abcdefghijklmnopqrstuvwxyz")
_DIGITS = set("0123456789")


def scan(text):
    """Return 'complete', 'prefix', or 'bad'."""
    index = 0
    size = len(text)

    def eat_literal(literal):
        nonlocal index
        for ch in literal:
            if index == size:
                return "need_more"
            if text[index] != ch:
                return "bad"
            index += 1
        return "ok"

    def eat_run(allowed):
        nonlocal index
        if index == size:
            return "need_more"
        if text[index] not in allowed:
            return "bad"
        while index < size and text[index] in allowed:
            index += 1
        if index == size:
            return "ended"
        return "ok"

    status = eat_literal("name=")
    if status != "ok":
        return "bad" if status == "bad" else "prefix"
    status = eat_run(_LETTERS)
    if status == "bad":
        return "bad"
    if status != "ok":
        return "prefix"
    status = eat_literal(" grade=")
    if status != "ok":
        return "bad" if status == "bad" else "prefix"
    status = eat_run(_DIGITS)
    if status == "bad":
        return "bad"
    if status != "ok":
        return "prefix"
    status = eat_literal(" club=")
    if status != "ok":
        return "bad" if status == "bad" else "prefix"
    status = eat_run(_LETTERS)
    if status == "bad":
        return "bad"
    if status == "need_more":
        return "prefix"
    if status == "ended":
        return "complete"
    return "bad"


def is_complete(text):
    return scan(text) == "complete"


def is_prefix(text):
    return scan(text) != "bad"


def fields(text):
    """Return (name, grade, club) or None if the line is not legal."""
    match = _LINE.fullmatch(text)
    if match is None:
        return None
    return match.group(1), int(match.group(2)), match.group(3)


def force_finish(text):
    """Add as few characters as possible so a prefix becomes a legal line."""
    # This order finishes a field before it lengthens a word.
    choices = "= 0123456789abcdefghijklmnopqrstuvwxyz"
    for _ in range(80):
        status = scan(text)
        if status == "complete":
            return text
        if status == "bad":
            return text
        added = None
        for ch in choices:
            if scan(text + ch) != "bad":
                added = ch
                break
        if added is None:
            return text
        text += added
    return text


def _check():
    good = "name=maya grade=10 club=robotics"
    assert scan(good) == "complete"
    assert fields(good) == ("maya", 10, "robotics")
    assert scan("") == "prefix"
    assert scan("name=") == "prefix"
    assert scan("name=maya") == "prefix"
    assert scan("name=maya grade=10") == "prefix"
    assert scan("name=maya grade=10 club=") == "prefix"
    assert scan("n") == "prefix"
    assert scan("name=Maya grade=10 club=robotics") == "bad"
    assert scan("name=maya  grade=10 club=robotics") == "bad"
    assert scan("name=maya grade=10 club=robotics ") == "bad"
    assert scan("name=maya grade=10 robotics") == "bad"
    assert fields("name=Maya grade=10 club=robotics") is None
    assert is_complete(force_finish(""))
    assert is_complete(force_finish("name=maya"))
    assert is_complete(force_finish("name=maya grade="))
    assert fields(force_finish("name=")) == ("a", 0, "a")
    assert scan("name=maya grade=10 club=robotics") == "complete"
    print("grammar checks passed")


if __name__ == "__main__":
    _check()
