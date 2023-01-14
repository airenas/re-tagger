import pytest

from src.utils.compare import drop_non_important


@pytest.mark.parametrize("w,mi,exp",
                         [
                             ("olia", "olia", "olia"),
                             ("olia", "Qg", "Qg"),
                             ("olia", "Sgg", "Sg-"),
                             ("olia", "Npmsgng", "Npmsgn-"),
                             ("olia", "Nnmsgn-", "Ncmsgn-"),
                             ("olia", "Nnmsgn-", "Ncmsgn-"),
                             ("olia", "Nxmsgn-", "Ncmsgn-"),
                             ("ir", "Q-", "Cg"),
                             ("olia", "Yg", "Y-"),
                             ("olia", "Xg", "X-"),
                         ])
class TestDropNonImportantTag:
    def test_drop_non_important(self, w, mi, exp):
        res = drop_non_important(w, mi)
        assert res == exp
