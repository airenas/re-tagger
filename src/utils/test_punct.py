import pytest

from src.utils.punct import is_punctuation


@pytest.mark.parametrize("f,exp",
                         [
                             ("m.", False),
                             (".", True),
                             ("...", True),
                             ("a", False),
                             ("1", False)
                         ])
class TestPunctuation:
    def test(self, f, exp):
        res = is_punctuation(f)
        assert res == exp
