import pytest

from src.utils.compare import drop_non_important


@pytest.mark.parametrize("mi,exp",
                         [
                             ("olia", "olia"),
                             ("Qg", "Q-"),
                             ("Sgg", "Sg-"),
                             ("Npmsgng", "Npmsgn-"),
                             ("Nnmsgn-", "Ncmsgn-"),
                             ("Nnmsgn-", "Ncmsgn-"),
                             ("Nxmsgn-", "Ncmsgn-"),
                         ])
class TestDropNonImportantTag:
    def test_drop_non_important(self, mi, exp):
        res = drop_non_important(mi)
        assert res == exp
