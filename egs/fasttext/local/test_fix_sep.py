import pytest

from egs.fasttext.local.fix_sep import fix_sep, one_upper_letter


@pytest.mark.parametrize("line,exp",
                         [
("end end.,.", "end end . , ."),
                            ("2020.04.16 2020.04", "2020 . 04 . 16 2020.04"),
                             ("2:4 a/b tada;", "2 : 4 a / b tada ;"),
                             ("-01,12 aaa", "-01,12 aaa"),
                             ("- x ,", "- x ,"),
                             ("A. B. G., Ž. I.. 0.", "A. B. G. , Ž. I. . 0 ."),
                             ("a ... b+* =", "a . . . b +* ="),
                             ("", ""),
                             ("word", "word"),
                             ("word,", "word ,"),
                             ("word, next.", "word , next ."),

                             ("end end .,.", "end end . , ."),
                             ("wow!?", "wow !?"),
                             ("- x ,", "- x ,"),
                             ("word,word", "word,word"),
                             ("0.12, a, b.", "0.12 , a , b ."),
                             ("a. ž.", "a. ž ."),
                             ("Pvz. 10 cnt.", "Pvz. 10 cnt."),
                             ("T.y. 10 cnt.", "T.y. 10 cnt."),
                             ("Nr. nr. aa", "Nr. nr. aa"),
                             ("A-BC-AA", "A - BC - AA"),
                             ("A-01-22", "A - 01 - 22"),
                             ("-01-22", "- 01 - 22"),
                             ("-1e-5 1e-5 3e+3", "-1e-5 1e-5 3e+3"),
                         ])
def test_fix_sep(line, exp):
    assert fix_sep(line) == exp


@pytest.mark.parametrize("line,exp",
                         [
                             ("A.", True),
                             ("I.", True),
                             ("0.", False),
                             ("A,", False),
                             ("AA.", False),
                             ("A.B.", False),
                             ("A.B", False),
                             ("A", False),
                             ("a.", True),
                             ("a.b.", False),
                             ("m.", True),
                             ("d.", True),
                             ("T.y.", True),
                             ("pvz.", True),
                             ("Pvz.", True),

                         ])
def test_one_upper_letter(line, exp):
    assert one_upper_letter(line) == exp
