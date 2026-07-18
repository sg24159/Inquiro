from shared.utils import strip_line_noise


def test_strip_line_noise_empty():
    assert strip_line_noise("") == ""


def test_strip_line_noise_whitespace_only():
    assert strip_line_noise("   ") == ""


def test_strip_line_noise_bullet_marker():
    assert strip_line_noise("- some text") == "some text"


def test_strip_line_noise_star_marker():
    assert strip_line_noise("* some text") == "some text"


def test_strip_line_noise_plus_marker():
    assert strip_line_noise("+ some text") == "some text"


def test_strip_line_noise_numbered_dot():
    assert strip_line_noise("1. some text") == "some text"


def test_strip_line_noise_numbered_paren():
    assert strip_line_noise("1) some text") == "some text"


def test_strip_line_noise_plain_text():
    assert strip_line_noise("no markers here") == "no markers here"


def test_strip_line_noise_strips_whitespace():
    assert strip_line_noise("  - indented  ") == "indented"
