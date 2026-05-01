from unittest.mock import MagicMock

from audio2subtitle.models import SubtitleEntry


class TestVocabFilter:
    def test_load_vocab_csvs(self, tmp_path):
        import pandas as pd

        for level in ["A1", "A2"]:
            df = pd.DataFrame(["hallo", "und", "ja"], columns=["word"])
            df.to_csv(tmp_path / f"{level}_vokabeln.csv", index=False, header=False)

        from audio2subtitle.filter import VocabFilter

        vocab = VocabFilter.load_vocab(str(tmp_path), ["A1", "A2"])
        assert "hallo" in vocab
        assert "und" in vocab
        assert len(vocab) == 3

    def test_load_vocab_missing_level(self, tmp_path):
        from audio2subtitle.filter import VocabFilter

        vocab = VocabFilter.load_vocab(str(tmp_path), ["Z99"])
        assert len(vocab) == 0

    def test_filter_keeps_advanced_segments(self):
        from audio2subtitle.filter import VocabFilter

        mock_nlp = MagicMock()
        mock_token1 = MagicMock()
        mock_token1.is_punct = False
        mock_token1.is_stop = False
        mock_token1.pos_ = "NOUN"
        mock_token1.lemma_ = "Quantenphysik"
        mock_nlp.return_value = [mock_token1]

        entry = SubtitleEntry(index=1, start_time=0.0, end_time=2.5, text="Quantenphysik ist komplex")
        result = VocabFilter.filter_segments([entry], mock_nlp, {"hallo", "und"})
        assert len(result) == 1

    def test_filter_removes_basic_segments(self):
        from audio2subtitle.filter import VocabFilter

        mock_nlp = MagicMock()
        mock_token = MagicMock()
        mock_token.is_punct = False
        mock_token.is_stop = False
        mock_token.pos_ = "NOUN"
        mock_token.lemma_ = "hallo"
        mock_nlp.return_value = [mock_token]

        entry = SubtitleEntry(index=1, start_time=0.0, end_time=2.5, text="hallo")
        result = VocabFilter.filter_segments([entry], mock_nlp, {"hallo", "und"})
        assert len(result) == 0

    def test_filter_skips_punct_stop_propn(self):
        from audio2subtitle.filter import VocabFilter

        mock_nlp = MagicMock()
        mock_token = MagicMock()
        mock_token.is_punct = True
        mock_token.is_stop = False
        mock_token.pos_ = "PUNCT"
        mock_token.lemma_ = "."
        mock_nlp.return_value = [mock_token]

        entry = SubtitleEntry(index=1, start_time=0.0, end_time=2.5, text=".")
        result = VocabFilter.filter_segments([entry], mock_nlp, {"hallo"})
        assert len(result) == 0
