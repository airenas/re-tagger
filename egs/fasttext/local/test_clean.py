import pytest

from egs.fasttext.local.clean import alnum_ratio, clean_text, has_punct_run, replace_punct, replace_url_email, skip


@pytest.mark.parametrize("w,exp",
                         [
                             ("jonas.j@delfi.lt", "<email>"),
                             ("jonas.j@delfi.lt,", "<email>,"),
                             ("https://www.delfi.lt/a?b=1.", "<url>."),
                             ("www.lrt.lt", "<url>"),
                             ("delfi.lt/x),", "<url>),"),
                             ("Vilnius,", "Vilnius,"),
                             ("m.Vilnius", "m.Vilnius"),
                             ("pvz.lt", "pvz.lt"),
                             ("a@b", "a@b"),
                         ])
def test_replace_url_email(w, exp):
    assert replace_url_email(w) == exp


@pytest.mark.parametrize("line,exp",
                         [
                             ("Labas...", False),
                             ("Labas....", True),
                             ("Kas?!?! ne", True),
                             ("Labas ... ...", False),
                             ("Ką (“...”) sakė", True),
                         ])
def test_has_punct_run(line, exp):
    assert has_punct_run(line) == exp


@pytest.mark.parametrize("line,exp",
                         [
                             ("abcd", 1.0),
                             ("ab cd", 1.0),
                             ("abc,", 0.75),
                             ("a1 b2", 1.0),
                             ("a+b", 2 / 3),
                             ("Ji.", 2 / 3),
                             ("", 0.0),
                             ("   ", 0.0),
                         ])
def test_alnum_ratio(line, exp):
    assert alnum_ratio(line) == pytest.approx(exp)


@pytest.mark.parametrize("line,exp",
                         [
                             ("abc, def, ghi", False),
                             ("abc, de", False),
                             ("abc,", False),
                             ("ab, cd", False),
                             ("abc, d, e,,", True),
                         ])
def test_skip_alnum_boundary(line, exp):
    assert skip(line) == exp


@pytest.mark.parametrize("line,exp",
                         [
                             ("Jis (labai) „puikus“ – taip; <url>, ok: 3+4*5/2 -x!?",
                              "Jis labai puikus - taip <url>, ok 3+4*5/2 -x!?"),
                             ("a – b — c 3−2", "a - b - c 3-2"),
                             ("a   b\t(c)", "a b c"),
                             ("a|b=c", "a|b=c"),
                         ])
def test_replace_punct(line, exp):
    assert replace_punct(line) == exp


@pytest.mark.parametrize("line,exp",
                         [
                             ("Labas pasauli, kaip sekasi!", False),
                             ("Ąžuolas „puikus“ 2024 m.", False),
                             ("Привет pasauli", True),
                             ("Köln yra miestas", True),
                             ("1234 - 56.", True),
                             ("...", True),
                             ("", True),
                             ("Ak", True),
                             ("jaaau labas", False),
                             ("jaaaau labas", True),
                             ("Aaaa bbb", True),
                             ("Labas | pasauli", True),
                             ("a, b, c, d, e, f", True),
                             ("Labas pasauli, kaip sekasi?", False),
                             ("ab, cd, e", False),
                             ("ab - c + d * e", True),
                             ("abc123 yra", True),
                             ("12abc yra", True),
                             ("pvz.lt yra", True),
                             ("t.y. taip", True),
                             ("Labas.Kitas", True),
                             ("kažkas-nors yra", False),
                             ("10-ųjų metų", False),
                             ("<url> ir <email> yra", False),
                             ("Labas " + "a" * 31 + "b", True),
                             ("Labas " + "ab" * 15, False),
                         ])
def test_skip(line, exp):
    assert skip(line) == exp


class TestCleanText:
    def test_strips_morphology_and_cleans(self):
        res, ok = clean_text("Jis(=Įv) labai(=Prv) „geras“ – taip.")
        assert ok
        assert res == "Jis labai geras - taip."

    def test_replaces_url_and_email(self):
        res, ok = clean_text("Rašykite jonas@delfi.lt arba https://www.delfi.lt/a ačiū")
        assert ok
        assert res == "Rašykite <email> arba <url> ačiū"

    def test_skips_non_lt_letters(self):
        _, ok = clean_text("Šis Köln miestas")
        assert not ok

    def test_skips_punct_run(self):
        _, ok = clean_text("Labas pasauli....")
        assert not ok
