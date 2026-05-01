import sys
from unittest.mock import MagicMock

import pysrt


def test_translate_srt(tmp_path, mocker):
    mock_ct2 = MagicMock()
    mock_transformers = MagicMock()
    mocker.patch.dict(sys.modules, {"ctranslate2": mock_ct2, "transformers": mock_transformers})

    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
    mock_ct2.Translator.return_value = mock_model

    mock_tokenizer.encode.return_value = [1, 2]
    mock_tokenizer.convert_ids_to_tokens.return_value = ["▁Hallo", "▁Welt"]
    mock_result = MagicMock()
    mock_result.hypotheses = [["▁Hola", "▁Mundo"]]
    mock_model.translate_batch.return_value = [mock_result]
    mock_tokenizer.convert_tokens_to_ids.return_value = [3, 4]
    mock_tokenizer.decode.return_value = "Hola mundo"

    from subtitle_translate.translator import MarianTranslator

    translator = MarianTranslator("/tmp/model", "Helsinki-NLP/opus-mt-tc-big-de-es")
    result = translator.translate_batch(["Hallo Welt"])
    assert result == ["Hola mundo"]


def test_translate_empty():
    from unittest.mock import patch, MagicMock

    with patch("subtitle_translate.translator.ctranslate2") as mock_ct2, \
         patch("subtitle_translate.translator.transformers") as mock_tf:
        mock_tf.AutoTokenizer.from_pretrained.return_value = MagicMock()
        mock_ct2.Translator.return_value = MagicMock()

        from subtitle_translate.translator import MarianTranslator
        translator = MarianTranslator("/tmp/model")
        assert translator.translate_batch([]) == []


def test_srt_io_roundtrip(tmp_path):
    import io
    import pysrt
    from subtitle_translate.srt_io import read_srt, write_srt

    subs = pysrt.SubRipFile()
    subs.append(pysrt.SubRipItem(index=1, start=pysrt.SubRipTime(seconds=0), end=pysrt.SubRipTime(seconds=2), text="Hola"))
    subs.append(pysrt.SubRipItem(index=2, start=pysrt.SubRipTime(seconds=2), end=pysrt.SubRipTime(seconds=4), text="Mundo"))

    out_file = tmp_path / "test.srt"
    write_srt(subs, str(out_file))

    loaded = read_srt(str(out_file))
    assert len(loaded) == 2
    assert loaded[0].text == "Hola"
    assert loaded[1].text == "Mundo"
