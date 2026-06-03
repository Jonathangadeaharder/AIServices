"""Text translation abstraction.

Provides:
- BaseTranslator ABC — callers should depend on this, not on concrete providers
- OpenAITranslator — cloud-based via OpenAI-compatible API
- NLLBTranslator — local via CTranslate2 (GPU-only, stub fallback)
- create_translator() factory — chooses provider via env var
- Convenience: translate_text()

Callers import from aiservices, never from openai / ctranslate2 directly.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

try:
    from structlog import get_logger
except ImportError:

    class Log:
        def info(self, *a, **k):
            pass
        def warning(self, *a, **k):
            pass

    def get_logger(n):
        return Log()


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Translation:
    """Result of a single translation operation."""
    text: str
    source_lang: str | None
    target_lang: str


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class BaseTranslator(ABC):
    """Abstract translator — all providers implement this.
    
    Callers depend on this interface, not concrete providers.
    """

    @abstractmethod
    def translate(
        self,
        text: str,
        target_lang: str = "English",
        *,
        source_lang: str | None = None,
    ) -> str:
        """Translate a single text string."""
        ...

    def translate_batch(
        self,
        texts: list[str],
        target_lang: str = "English",
        *,
        source_lang: str | None = None,
    ) -> list[str]:
        """Translate multiple texts."""
        return [
            self.translate(t, target_lang, source_lang=source_lang)
            for t in texts
        ]


# ---------------------------------------------------------------------------
# Concrete providers
# ---------------------------------------------------------------------------


class OpenAITranslator(BaseTranslator):
    """Cloud translation via OpenAI-compatible API.
    
    Uses JUDGE_API_URL / JUDGE_MODEL / JUDGE_API_KEY env vars.
    Default: localhost:8000/v1 with nemotron-omni.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self._model = model or os.getenv("JUDGE_MODEL", "nemotron-omni")
        self._base_url = base_url or os.getenv("JUDGE_API_URL", "http://localhost:8000/v1")
        self._api_key = api_key or os.getenv("JUDGE_API_KEY", "local")

    def translate(
        self,
        text: str,
        target_lang: str = "English",
        *,
        source_lang: str | None = None,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(base_url=self._base_url, api_key=self._api_key)

        source = f" from {source_lang}" if source_lang else ""
        system = (
            f"You are a professional translator. "
            f"Translate the user's text{source} to {target_lang}. "
            f"Return only the translation, no explanation."
        )

        logger.info(
            "openai_translate_start",
            chars=len(text),
            target=target_lang,
            model=self._model,
        )
        resp = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            max_tokens=len(text) * 3,
            temperature=0.1,
        )
        translated = resp.choices[0].message.content.strip()
        logger.info("openai_translate_done", chars=len(translated))
        return translated

    def translate_batch(
        self,
        texts: list[str],
        target_lang: str = "English",
        *,
        source_lang: str | None = None,
    ) -> list[str]:
        from openai import OpenAI

        client = OpenAI(base_url=self._base_url, api_key=self._api_key)

        source = f" from {source_lang}" if source_lang else ""
        system = (
            f"You are a professional translator. "
            f"Translate the following texts{source} to {target_lang}. "
            f"Return only the translations, one per line, preserving order. "
            f"When input is numbered, strip the numbers — return plain lines."
        )
        user_content = "\n".join(texts)

        logger.info(
            "openai_translate_batch_start",
            count=len(texts),
            target=target_lang,
            model=self._model,
        )
        resp = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            max_tokens=sum(len(t) for t in texts) * 3,
            temperature=0.1,
        )
        translated = resp.choices[0].message.content
        if translated is None:
            return [""] * len(texts)

        results = [
            line.strip().lstrip("-*0123456789. \t")
            for line in translated.strip().split("\n")
            if line.strip()
        ]

        if len(results) != len(texts):
            logger.warning(
                "translation_count_mismatch",
                expected=len(texts),
                received=len(results),
            )
            if len(results) < len(texts):
                results.extend([""] * (len(texts) - len(results)))
            else:
                results = results[: len(texts)]

        return results


class NLLBTranslator(BaseTranslator):
    """Local translation via CTranslate2 NLLB model.
    
    Requires ctranslate2 (GPU-only dependency, not always available).
    Falls back to error when not installed.
    """

    def __init__(self, device: str = "cpu"):
        self._device = device

    def translate(
        self,
        text: str,
        target_lang: str = "English",
        *,
        source_lang: str | None = None,
    ) -> str:
        return self.translate_batch([text], target_lang, source_lang=source_lang)[0]

    def translate_batch(
        self,
        texts: list[str],
        target_lang: str = "English",
        *,
        source_lang: str | None = None,
    ) -> list[str]:
        try:
            import ctranslate2  # type: ignore[import]  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "ctranslate2 not installed — NLLB requires ctranslate2.\n"
                "Install: pip install ctranslate2\n"
                "Or use OpenAITranslator instead."
            ) from e

        # NLLB uses 3-letter codes (eng_Latn, deu_Latn, fra_Latn, etc.)
        source_code = _lang_to_nllb(source_lang) if source_lang else None
        target_code = _lang_to_nllb(target_lang)

        logger.info(
            "nllb_translate_batch_start",
            count=len(texts),
            source=source_code,
            target=target_code,
            device=self._device,
        )

        translator = ctranslate2.Translator("nllb-200-distilled-600M", device=self._device)
        tokenizer_file = "flores200_sentencepiece.model"

        results: list[str] = []
        for text in texts:
            import sentencepiece as spm  # type: ignore[import]

            sp = spm.SentencePieceProcessor()
            sp.load(tokenizer_file)

            source_tokens = sp.encode(text, out_type=str)
            if target_code:
                target_prefix = [target_code]
            else:
                target_prefix = []
            translated_tokens = translator.translate(
                [source_tokens],
                target_prefix=target_prefix,
            )[0].hypotheses[0]
            translated_text = sp.decode(translated_tokens)
            results.append(translated_text)

        return results


# ---------------------------------------------------------------------------
# Language code helpers
# ---------------------------------------------------------------------------


def _lang_to_nllb(lang: str) -> str:
    """Convert language name or code to NLLB format (e.g. eng_Latn)."""
    # Common mappings
    mapping = {
        "english": "eng_Latn",
        "german": "deu_Latn",
        "french": "fra_Latn",
        "spanish": "spa_Latn",
        "italian": "ita_Latn",
        "portuguese": "por_Latn",
        "dutch": "nld_Latn",
        "russian": "rus_Cyrl",
        "chinese": "zho_Hans",
        "japanese": "jpn_Jpan",
        "korean": "kor_Hang",
        "arabic": "arb_Arab",
        "hindi": "hin_Deva",
        "turkish": "tur_Latn",
        "polish": "pol_Latn",
        "swedish": "swe_Latn",
        "danish": "dan_Latn",
        "norwegian": "nno_Latn",
        "finnish": "fin_Latn",
        "czech": "ces_Latn",
        "romanian": "ron_Latn",
        "hungarian": "hun_Latn",
        "greek": "ell_Grek",
        "hebrew": "heb_Hebr",
        "thai": "tha_Thai",
        "vietnamese": "vie_Latn",
        "indonesian": "ind_Latn",
        "malay": "msa_Latn",
        "tagalog": "tgl_Latn",
        "ukrainian": "ukr_Cyrl",
    }
    key = lang.lower().replace("-", "_")
    if key in mapping:
        return mapping[key]
    # Already in NLLB format?
    if "_" in lang and len(lang) <= 12:
        return lang
    # Default fallback
    logger.warning("unknown_language_for_nllb", lang=lang)
    return lang


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_TRANSLATION_PROVIDER = os.getenv("TRANSLATION_PROVIDER", "auto")


def create_translator(provider: str | None = None) -> BaseTranslator:
    """Create a translator instance.

    Args:
        provider: One of "openai", "nllb", "local", "auto" (default).
            Auto tries NLLB first, falls back to OpenAI.

    Returns:
        A BaseTranslator instance.

    Raises:
        RuntimeError: If no provider available.
    """
    name = provider or _TRANSLATION_PROVIDER

    if name == "auto":
        try:
            import ctranslate2  # type: ignore[import]  # noqa: F401
            return NLLBTranslator()
        except ImportError:
            pass
        try:
            from openai import OpenAI  # type: ignore[import]  # noqa: F401
            return OpenAITranslator()
        except ImportError:
            pass
        raise RuntimeError(
            "No translation backend available. "
            "Install ctranslate2 (GPU) or ensure openai SDK is available."
        )

    if name in ("openai", "gpt", "api"):
        return OpenAITranslator()

    if name in ("nllb", "local", "ct2"):
        return NLLBTranslator()

    raise ValueError(f"Unknown translation provider: {name!r}")


# ---------------------------------------------------------------------------
# Convenience functions (backward compatible)
# ---------------------------------------------------------------------------


def translate_text(
    text: str,
    target_lang: str = "English",
    *,
    source_lang: str | None = None,
    provider: str | None = None,
) -> str:
    """Translate text.

    Args:
        text: Text to translate
        target_lang: Target language (name or code)
        source_lang: Optional source language
        provider: Backend provider ("openai", "nllb", "local", "auto")

    Returns:
        Translated text
    """
    translator = create_translator(provider)
    return translator.translate(text, target_lang, source_lang=source_lang)
