import sys
from unittest.mock import MagicMock


class TestMarianTranslator:
    def test_translate_batch_calls_model(self, mocker):
        mock_ct2 = MagicMock()
        mock_transformers = MagicMock()
        mocker.patch.dict(sys.modules, {"ctranslate2": mock_ct2, "transformers": mock_transformers})

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_ct2.Translator.return_value = mock_model

        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.convert_ids_to_tokens.return_value = ["▁Hallo", "▁Welt"]

        mock_result = MagicMock()
        mock_result.hypotheses = [["▁Hola", "▁Mundo"]]
        mock_model.translate_batch.return_value = [mock_result]

        mock_tokenizer.convert_tokens_to_ids.return_value = [4, 5]
        mock_tokenizer.decode.return_value = "Hola mundo"

        from audio2subtitle.translator import MarianTranslator

        translator = MarianTranslator("Helsinki-NLP/opus-mt-tc-big-de-es", "/tmp/model")
        result = translator.translate_batch(["Hallo Welt"])

        assert result == ["Hola mundo"]
        mock_ct2.Translator.assert_called_once_with("/tmp/model")
        mock_model.translate_batch.assert_called_once()

    def test_translate_batch_empty_list(self, mocker):
        mock_ct2 = MagicMock()
        mock_transformers = MagicMock()
        mocker.patch.dict(sys.modules, {"ctranslate2": mock_ct2, "transformers": mock_transformers})
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()
        mock_ct2.Translator.return_value = MagicMock()

        from audio2subtitle.translator import MarianTranslator

        translator = MarianTranslator("Helsinki-NLP/opus-mt-tc-big-de-es", "/tmp/model")
        result = translator.translate_batch([])
        assert result == []

    def test_translate_batch_processes_in_chunks(self, mocker):
        mock_ct2 = MagicMock()
        mock_transformers = MagicMock()
        mocker.patch.dict(sys.modules, {"ctranslate2": mock_ct2, "transformers": mock_transformers})

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_ct2.Translator.return_value = mock_model

        mock_tokenizer.encode.return_value = [1]
        mock_tokenizer.convert_ids_to_tokens.return_value = ["▁x"]

        mock_result = MagicMock()
        mock_result.hypotheses = [["▁y"]]
        mock_model.translate_batch.return_value = [mock_result]

        mock_tokenizer.convert_tokens_to_ids.return_value = [2]
        mock_tokenizer.decode.return_value = "y"

        from audio2subtitle.translator import MarianTranslator

        translator = MarianTranslator("Helsinki-NLP/opus-mt-tc-big-de-es", "/tmp/model")
        texts = [f"text{i}" for i in range(35)]
        result = translator.translate_batch(texts, batch_size=16)

        assert len(result) == 35
        assert mock_model.translate_batch.call_count == 2
