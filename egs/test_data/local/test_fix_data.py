import pytest

from egs.test_data.local.fix_data import is_number_with_dash, replace_number_with_dash


@pytest.mark.parametrize("w,exp",
                         [
                             ("olia", False),
                             ("10-ųjų", True),
                             ("olia-ųjų", False),
                         ])
class TestFindNumber:
    def test_number_with_dash(self, w, exp):
        res = is_number_with_dash(w)
        assert res == exp


@pytest.mark.parametrize("w,exp",
                         [
                             ("olia", "olia"),
                             ("10-ųjų", "n-ųjų"),
                             ("olia-ųjų", "olia-ųjų"),
                             ("3-ųjų", "n-ųjų"),
                             ("1990-ieji", "n-ieji"),
                         ])
class TestReplaceNumber:
    def test_number_with_dash(self, w, exp):
        res = replace_number_with_dash(w)
        assert res == exp
