from unittest.mock import MagicMock

import pysrt


def test_load_vocab(tmp_path):
    import pandas as pd
    from subtitle_filter.filter import VocabFilter

    for level in ["A1", "A2"]:
        df = pd.DataFrame(["hallo", "und", "ja"], columns=["word"])
        df.to_csv(tmp_path / f"{level}_vokabeln.csv", index=False, header=False)

    vocab = VocabFilter.load_vocab(str(tmp_path), ["A1", "A2"])
    assert "hallo" in vocab
    assert len(vocab) == 3


def test_load_vocab_missing_level(tmp_path):
    from subtitle_filter.filter import VocabFilter
    assert VocabFilter.load_vocab(str(tmp_path), ["Z99"]) == set()


def test_filter_keeps_advanced():
    from subtitle_filter.filter import VocabFilter

    mock_nlp = MagicMock()
    mock_token = MagicMock()
    mock_token.is_punct = False
    mock_token.is_stop = False
    mock_token.pos_ = "NOUN"
    mock_token.lemma_ = "Quantenphysik"
    mock_nlp.return_value = [mock_token]

    subs = pysrt.SubRipFile()
    subs.append(pysrt.SubRipItem(index=1, start=pysrt.SubRipTime(seconds=0), end=pysrt.SubRipTime(seconds=2), text="Quantenphysik ist komplex"))

    result = VocabFilter.filter_subs(subs, mock_nlp, {"hallo"})
    assert len(result) == 1


def test_filter_removes_basic():
    from subtitle_filter.filter import VocabFilter

    mock_nlp = MagicMock()
    mock_token = MagicMock()
    mock_token.is_punct = False
    mock_token.is_stop = False
    mock_token.pos_ = "NOUN"
    mock_token.lemma_ = "hallo"
    mock_nlp.return_value = [mock_token]

    subs = pysrt.SubRipFile()
    subs.append(pysrt.SubRipItem(index=1, start=pysrt.SubRipTime(seconds=0), end=pysrt.SubRipTime(seconds=2), text="hallo"))

    result = VocabFilter.filter_subs(subs, mock_nlp, {"hallo", "und"})
    assert len(result) == 0


def test_filter_reindexes():
    from subtitle_filter.filter import VocabFilter

    mock_nlp = MagicMock()
    mock_token = MagicMock()
    mock_token.is_punct = False
    mock_token.is_stop = False
    mock_token.pos_ = "NOUN"
    mock_token.lemma_ = "advanced"
    mock_nlp.return_value = [mock_token]

    subs = pysrt.SubRipFile()
    subs.append(pysrt.SubRipItem(index=1, start=pysrt.SubRipTime(seconds=0), end=pysrt.SubRipTime(seconds=2), text="advanced"))
    subs.append(pysrt.SubRipItem(index=2, start=pysrt.SubRipTime(seconds=2), end=pysrt.SubRipTime(seconds=4), text="advanced"))

    result = VocabFilter.filter_subs(subs, mock_nlp, {"hallo"})
    assert result[0].index == 1
    assert result[1].index == 2
