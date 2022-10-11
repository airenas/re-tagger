import pytest

from egs.crf.local.restore import restore


@pytest.mark.parametrize("all_tags,pred,tags,exp",
                         [
                             ("Agpn--n:Agpfsvn:Agpfsin:Agpfsnn", "A--f---", {}, "Agpfsvn"),
                             ("Agpn--n:Agpfsvn:Agpfsin:Agpfsnn", "A--f---", {"Agpfsin": 1}, "Agpfsin"),
                             ("Agpn--n:Agpfsvn:Agpfsin:Agpfsnn", "A--f---", {"Agpfsin": 1, "Agpfsnn": 2000}, "Agpfsnn"),
                         ])
class TestRound:
    def test_round(self, all_tags, pred, tags, exp):
        res, _, _, _ = restore(all_tags.split(":"), pred, tags)
        assert res == exp
