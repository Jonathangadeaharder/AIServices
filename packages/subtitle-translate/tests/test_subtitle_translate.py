import sys
from unittest.mock import MagicMock


def test_translate_srt(tmp_path, monkeypatch):
    mock_ct2 = MagicMock()
    mock_transformers = MagicMock()
    monkeypatch.setitem(sys.modules, "ctranslate2", mock_ct2)
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

    sys.modules.pop("subtitle_translate.translator", None)

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


def test_translate_batch_empty(tmp_path, monkeypatch):
    mock_ct2 = MagicMock()
    mock_transformers = MagicMock()
    monkeypatch.setitem(sys.modules, "ctranslate2", mock_ct2)
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

    sys.modules.pop("subtitle_translate.translator", None)

    mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()
    mock_ct2.Translator.return_value = MagicMock()

    from subtitle_translate.translator import MarianTranslator

    translator = MarianTranslator("/tmp/model")
    assert translator.translate_batch([]) == []


def test_srt_io_roundtrip(tmp_path):
    from subtitle_translate.srt_io import read_srt

    srt_content = """\
1
00:00:00,000 --> 00:00:02,000
Hola

2
00:00:02,000 --> 00:00:04,000
Mundo
"""
    srt_file = tmp_path / "test.srt"
    srt_file.write_text(srt_content, encoding="utf-8")

    loaded = read_srt(str(srt_file))
    assert len(loaded) == 2
    assert loaded[0].text == "Hola"
    assert loaded[1].text == "Mundo"
