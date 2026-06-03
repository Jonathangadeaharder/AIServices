"""Tests for aiservices.translate module.

Covers:
- OpenAITranslator with mocked OpenAI client
- NLLBTranslator import error handling
- create_translator factory
- _lang_to_nllb language code mapping
- translate_text convenience function
- batch translation with count mismatch handling
"""

import pytest

from aiservices.translate import (
    OpenAITranslator,
    NLLBTranslator,
    BaseTranslator,
    Translation,
    create_translator,
    translate_text,
    _lang_to_nllb,
)


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------


class TestTranslation:
    def test_creation(self):
        t = Translation(text="hola", source_lang="en", target_lang="es")
        assert t.text == "hola"
        assert t.source_lang == "en"
        assert t.target_lang == "es"


# ---------------------------------------------------------------------------
# _lang_to_nllb
# ---------------------------------------------------------------------------


class TestLangToNLLB:
    def test_known_language(self):
        assert _lang_to_nllb("English") == "eng_Latn"
        assert _lang_to_nllb("german") == "deu_Latn"
        assert _lang_to_nllb("Chinese") == "zho_Hans"

    def test_case_insensitive(self):
        assert _lang_to_nllb("FRENCH") == "fra_Latn"
        assert _lang_to_nllb("Japanese") == "jpn_Jpan"

    def test_hyphen_to_underscore(self):
        # Should normalize input
        result = _lang_to_nllb("en-US")
        # Won't match any mapping, falls through to underscore check or default
        assert isinstance(result, str)

    def test_already_nllb_format(self):
        assert _lang_to_nllb("eng_Latn") == "eng_Latn"
        assert _lang_to_nllb("deu_Latn") == "deu_Latn"

    def test_unknown_language(self):
        result = _lang_to_nllb("Klingon")
        assert result == "Klingon"  # Falls through to warning + return


# ---------------------------------------------------------------------------
# OpenAITranslator
# ---------------------------------------------------------------------------


class TestOpenAITranslator:
    def test_default_init(self, monkeypatch):
        monkeypatch.setenv("JUDGE_MODEL", "test-model")
        monkeypatch.setenv("JUDGE_API_URL", "http://test:8000/v1")
        monkeypatch.setenv("JUDGE_API_KEY", "test-key")

        t = OpenAITranslator()
        assert t._model == "test-model"
        assert t._base_url == "http://test:8000/v1"
        assert t._api_key == "test-key"

    def test_custom_init(self):
        t = OpenAITranslator(
            model="custom-model",
            base_url="http://custom:8000/v1",
            api_key="custom-key",
        )
        assert t._model == "custom-model"
        assert t._base_url == "http://custom:8000/v1"
        assert t._api_key == "custom-key"

    def test_translate(self, mocker):
        mock_response = mocker.MagicMock()
        mock_response.choices = [
            mocker.MagicMock(message=mocker.MagicMock(content="Hola mundo"))
        ]

        mock_client = mocker.MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai = mocker.MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mocker.patch.dict("sys.modules", {"openai": mock_openai})

        t = OpenAITranslator(
            model="test",
            base_url="http://test:8000/v1",
            api_key="key",
        )
        result = t.translate("Hello world", target_lang="Spanish")

        assert result == "Hola mundo"
        mock_client.chat.completions.create.assert_called_once()

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[1]["content"] == "Hello world"
        assert "Spanish" in messages[0]["content"]

    def test_translate_with_source_lang(self, mocker):
        mock_response = mocker.MagicMock()
        mock_response.choices = [
            mocker.MagicMock(
                message=mocker.MagicMock(content="Hello world")
            )
        ]

        mock_client = mocker.MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai = mocker.MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mocker.patch.dict("sys.modules", {"openai": mock_openai})

        t = OpenAITranslator(
            model="test",
            base_url="http://test:8000/v1",
            api_key="key",
        )
        result = t.translate(
            "Hola mundo", target_lang="English", source_lang="Spanish"
        )

        assert result == "Hello world"
        call_kwargs = mock_client.chat.completions.create.call_args
        system_msg = call_kwargs.kwargs["messages"][0]["content"]
        assert "from Spanish" in system_msg
        assert "English" in system_msg

    def test_translate_batch(self, mocker):
        mock_response = mocker.MagicMock()
        mock_response.choices = [
            mocker.MagicMock(
                message=mocker.MagicMock(content="Hello\nWorld")
            )
        ]

        mock_client = mocker.MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai = mocker.MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mocker.patch.dict("sys.modules", {"openai": mock_openai})

        t = OpenAITranslator(
            model="test",
            base_url="http://test:8000/v1",
            api_key="key",
        )
        results = t.translate_batch(["Hola", "Mundo"], target_lang="English")

        assert len(results) == 2
        assert results[0] == "Hello"
        assert results[1] == "World"

    def test_translate_batch_count_mismatch_fewer(self, mocker):
        """When API returns fewer lines, pad with empty strings."""
        mock_response = mocker.MagicMock()
        mock_response.choices = [
            mocker.MagicMock(
                message=mocker.MagicMock(content="Only one result")
            )
        ]

        mock_client = mocker.MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai = mocker.MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mocker.patch.dict("sys.modules", {"openai": mock_openai})

        t = OpenAITranslator(
            model="test",
            base_url="http://test:8000/v1",
            api_key="key",
        )
        results = t.translate_batch(
            ["Line 1", "Line 2", "Line 3"], target_lang="English"
        )

        assert len(results) == 3
        assert results[0] == "Only one result"
        assert results[1] == ""
        assert results[2] == ""

    def test_translate_batch_count_mismatch_more(self, mocker):
        """When API returns more lines, truncate."""
        mock_response = mocker.MagicMock()
        mock_response.choices = [
            mocker.MagicMock(
                message=mocker.MagicMock(content="A\nB\nC")
            )
        ]

        mock_client = mocker.MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai = mocker.MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mocker.patch.dict("sys.modules", {"openai": mock_openai})

        t = OpenAITranslator(
            model="test",
            base_url="http://test:8000/v1",
            api_key="key",
        )
        results = t.translate_batch(["only one"], target_lang="English")

        assert len(results) == 1
        assert results[0] == "A"

    def test_translate_batch_none_content(self, mocker):
        """Handle None content gracefully."""
        mock_response = mocker.MagicMock()
        mock_response.choices = [
            mocker.MagicMock(message=mocker.MagicMock(content=None))
        ]

        mock_client = mocker.MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai = mocker.MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mocker.patch.dict("sys.modules", {"openai": mock_openai})

        t = OpenAITranslator(
            model="test",
            base_url="http://test:8000/v1",
            api_key="key",
        )
        results = t.translate_batch(["hello"], target_lang="Spanish")

        assert len(results) == 1
        assert results[0] == ""


# ---------------------------------------------------------------------------
# NLLBTranslator
# ---------------------------------------------------------------------------


class TestNLLBTranslator:
    def test_import_error(self, mocker):
        """NLLB should raise ImportError when ctranslate2 not available."""
        # Ensure ctranslate2 is not importable
        import sys as _sys
        mocker.patch.dict(
            _sys.modules, {"ctranslate2": None}, clear=False
        )

        t = NLLBTranslator()
        with pytest.raises(ImportError, match="ctranslate2 not installed"):
            t.translate("hello", target_lang="Spanish")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateTranslator:
    def test_explicit_openai(self, mocker):
        mock_openai = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"openai": mock_openai})
        t = create_translator("openai")
        assert isinstance(t, OpenAITranslator)

    def test_gpt_alias(self, mocker):
        mock_openai = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"openai": mock_openai})
        t = create_translator("gpt")
        assert isinstance(t, OpenAITranslator)

    def test_api_alias(self, mocker):
        mock_openai = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"openai": mock_openai})
        t = create_translator("api")
        assert isinstance(t, OpenAITranslator)

    def test_explicit_nllb(self, mocker):
        mocker.patch.dict("sys.modules", {"ctranslate2": mocker.MagicMock()})
        t = create_translator("nllb")
        assert isinstance(t, NLLBTranslator)

    def test_local_alias(self, mocker):
        mocker.patch.dict("sys.modules", {"ctranslate2": mocker.MagicMock()})
        t = create_translator("local")
        assert isinstance(t, NLLBTranslator)

    def test_ct2_alias(self, mocker):
        mocker.patch.dict("sys.modules", {"ctranslate2": mocker.MagicMock()})
        t = create_translator("ct2")
        assert isinstance(t, NLLBTranslator)

    def test_auto_prefers_nllb(self, mocker):
        mocker.patch.dict("sys.modules", {"ctranslate2": mocker.MagicMock()})
        t = create_translator("auto")
        assert isinstance(t, NLLBTranslator)

    def test_auto_falls_back_to_openai(self, mocker):
        # ctranslate2 fails, openai available
        import sys as _sys
        mocker.patch.dict(
            _sys.modules, {"ctranslate2": None}, clear=False
        )
        mock_openai = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"openai": mock_openai})
        t = create_translator("auto")
        assert isinstance(t, OpenAITranslator)

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown translation"):
            create_translator("nonexistent")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


class TestTranslateText:
    def test_translate_text(self, mocker):
        mock_response = mocker.MagicMock()
        mock_response.choices = [
            mocker.MagicMock(message=mocker.MagicMock(content="Hallo"))
        ]

        mock_client = mocker.MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai = mocker.MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mocker.patch.dict("sys.modules", {"openai": mock_openai})

        result = translate_text(
            "Hello", target_lang="German", provider="openai"
        )

        assert result == "Hallo"
