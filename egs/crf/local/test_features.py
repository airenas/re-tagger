from egs.crf.local.features import word2features, feat_word_v1


class TestFeatures:
    def test_bos(self):
        res = word2features([['olia', '', ''], ['olia', '', '']], 0,
                            {"words_before": 0, "words_after": 0, "skip_punct": False},
                            feat_word_v1)
        assert res["BOS"]
        res = word2features([['olia', '', ''], ['olia', '', '']], 1,
                            {"words_before": 0, "words_after": 0, "skip_punct": False},
                            feat_word_v1)
        assert res.get("BOS", False) == False

    def test_punct(self):
        res = word2features([['olia', '', ''], ['w1', '', ''], ['.', '', '']], 0,
                            {"words_before": 0, "words_after": 2, "skip_punct": False},
                            feat_word_v1)
        assert res["word"] == 'olia'
        assert res["+1:word"] == 'w1'
        assert res["+2:word"] == '.'

        res = word2features([['olia', '', ''], ['w1', '', ''], ['.', '', ''], ['w2', '', '']], 0,
                            {"words_before": 0, "words_after": 2, "skip_punct": True},
                            feat_word_v1)
        assert res["+1:word"] == 'w1'
        assert res["+2:word"] == 'w2'

    def test_punct_before(self):
        res = word2features([['olia', '', ''], ['w1', '', ''], ['.', '', ''], ['w2', '', '']], 3,
                            {"words_before": 2, "words_after": 2, "skip_punct": False},
                            feat_word_v1)
        assert res["-1:word"] == '.'
        assert res["-2:word"] == 'w1'

        res = word2features([['olia', '', ''], ['w1', '', ''], ['.', '', ''], ['w2', '', '']], 3,
                            {"words_before": 2, "words_after": 2, "skip_punct": True},
                            feat_word_v1)
        assert res["-1:word"] == 'w1'
        assert res["-2:word"] == 'olia'
