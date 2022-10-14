import pytest

from src.utils.conllu import extract_tag


@pytest.mark.parametrize("f,exp",
                         [
                             ("olia", "olia"),
                             ("Multext=olia", "olia"),
                         ])
class TestExtractTag:
    def test_extraxt(self, f, exp):
        res = extract_tag(f)
        assert res == exp
