# audio2subtitle: Vocabulary Filtering + Translation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `audio2subtitle` with optional vocabulary filtering (spaCy + German A1/A2/B1 CSVs) and translation (MarianMT DE→ES) to replicate the Colab subtitle pipeline, translated-text-only SRT output.

**Architecture:** Add `filter.py` and `translator.py` modules to the `audio2subtitle` package. Both are optional post-processing steps triggered by new request fields. The MLX provider calls them after whisper transcription when enabled. Translation uses CTranslate2 (fast CPU inference, no GPU needed). spaCy runs on CPU.

**Tech Stack:** mlx-whisper (existing), ctranslate2, transformers (tokenizer only), spacy, pandas, sentencepiece

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `packages/audio2subtitle/src/audio2subtitle/models.py` | Modify | Add `translated_text` to `SubtitleEntry`, add filter/translate fields to `Request` |
| `packages/audio2subtitle/src/audio2subtitle/filter.py` | Create | `VocabFilter` — loads CSVs, spaCy lemmatization, segment filtering |
| `packages/audio2subtitle/src/audio2subtitle/translator.py` | Create | `MarianTranslator` — loads MarianMT, batch translate |
| `packages/audio2subtitle/src/audio2subtitle/providers/mlx.py` | Modify | Wire filter + translate into generate() |
| `packages/audio2subtitle/src/audio2subtitle/cli.py` | Modify | Add `--translate-to`, `--vocab-filter`, `--vocab-levels` options |
| `packages/audio2subtitle/pyproject.toml` | Modify | Add optional deps |
| `packages/audio2subtitle/tests/test_filter.py` | Create | Unit tests for VocabFilter |
| `packages/audio2subtitle/tests/test_translator.py` | Create | Unit tests for MarianTranslator |
| `packages/audio2subtitle/tests/test_audio2subtitle.py` | Modify | Add tests for filter+translate in provider |

---

### Task 1: Extend Models

**Files:**
- Modify: `packages/audio2subtitle/src/audio2subtitle/models.py`
- Modify: `packages/audio2subtitle/tests/test_audio2subtitle.py`

- [ ] **Step 1: Write the failing test for new model fields**

```python
# Add to packages/audio2subtitle/tests/test_audio2subtitle.py

class TestTranslatedEntry:
    def test_subtitle_entry_translated_text(self):
        entry = SubtitleEntry(index=1, start_time=0.0, end_time=2.5, text="Hallo Welt", translated_text="Hola Mundo")
        assert entry.translated_text == "Hola Mundo"

    def test_subtitle_entry_translated_text_default_none(self):
        entry = SubtitleEntry(index=1, start_time=0.0, end_time=2.5, text="Hello")
        assert entry.translated_text is None

    def test_request_translate_to(self):
        req = Audio2SubtitleRequest(audio_path="/tmp/audio.wav", translate_to="es")
        assert req.translate_to == "es"
        assert req.translation_model == "Helsinki-NLP/opus-mt-tc-big-de-es"

    def test_request_vocab_filter(self):
        req = Audio2SubtitleRequest(audio_path="/tmp/audio.wav", vocab_filter_path="/data/vocab", vocab_levels=["A1", "B1"])
        assert req.vocab_filter_path == "/data/vocab"
        assert req.vocab_levels == ["A1", "B1"]

    def test_request_defaults_no_filter_no_translate(self):
        req = Audio2SubtitleRequest(audio_path="/tmp/audio.wav")
        assert req.translate_to is None
        assert req.vocab_filter_path is None
        assert req.vocab_levels == ["A1", "A2", "B1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/audio2subtitle/tests/test_audio2subtitle.py -k "TestTranslatedEntry" -v`
Expected: FAIL — `translated_text` field doesn't exist, `translate_to`/`vocab_filter_path` fields don't exist

- [ ] **Step 3: Implement model changes**

```python
# packages/audio2subtitle/src/audio2subtitle/models.py

from typing import Any, Literal
from pydantic import BaseModel, Field


class Audio2SubtitleRequest(BaseModel):
    audio_path: str = Field(..., description="Path to the input audio file")
    language: str | None = Field(None, description="Language of the audio (auto-detect if None)")
    output_format: Literal["srt", "vtt"] = Field("srt", description="Output format: srt or vtt")
    model_name: str = Field(
        "mlx-community/whisper-large-v3-turbo",
        description="Whisper model name",
    )
    word_timestamps: bool = Field(True, description="Enable word-level timestamps")
    translate_to: str | None = Field(
        None, description="Target language code for translation (e.g. 'es'). None = no translation."
    )
    translation_model: str = Field(
        "Helsinki-NLP/opus-mt-tc-big-de-es",
        description="HuggingFace translation model name (for tokenizer)",
    )
    ct2_model_path: str = Field(
        "",
        description="Path to CTranslate2-converted model directory. Empty = auto-convert on first use.",
    )
    vocab_filter_path: str | None = Field(
        None,
        description="Path to directory containing vocab CSVs (A1_vokabeln.csv, etc.). None = no filtering.",
    )
    vocab_levels: list[str] = Field(
        default_factory=lambda: ["A1", "A2", "B1"],
        description="Vocabulary levels to load for filtering",
    )


class SubtitleEntry(BaseModel):
    index: int = Field(..., description="Subtitle entry index (1-based)")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Subtitle text")
    translated_text: str | None = Field(None, description="Translated subtitle text")


class Audio2SubtitleResponse(BaseModel):
    output_path: str = Field(..., description="Path to the output subtitle file")
    entries: list[SubtitleEntry] = Field(default_factory=list, description="Subtitle entries")
    language: str | None = Field(None, description="Detected language")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/audio2subtitle/tests/test_audio2subtitle.py -k "TestTranslatedEntry" -v`
Expected: PASS

- [ ] **Step 5: Run all existing tests to verify no regressions**

Run: `uv run pytest packages/audio2subtitle/tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add packages/audio2subtitle/src/audio2subtitle/models.py packages/audio2subtitle/tests/test_audio2subtitle.py
git commit -m "feat(audio2subtitle): add translated_text, translate_to, vocab_filter fields to models"
```

---

### Task 2: Vocabulary Filter Module

**Files:**
- Create: `packages/audio2subtitle/src/audio2subtitle/filter.py`
- Create: `packages/audio2subtitle/tests/test_filter.py`

- [ ] **Step 1: Write the failing tests**

```python
# packages/audio2subtitle/tests/test_filter.py

import pytest
from unittest.mock import MagicMock


class TestVocabFilter:
    def test_load_vocab_csvs(self, tmp_path):
        import pandas as pd
        # Create mock CSV files
        for level in ["A1", "A2"]:
            df = pd.DataFrame(["hallo", "und", "ja"], columns=["word"])
            df.to_csv(tmp_path / f"{level}_vokabeln.csv", index=False, header=False)

        from audio2subtitle.filter import VocabFilter
        vocab = VocabFilter.load_vocab(str(tmp_path), ["A1", "A2"])
        assert "hallo" in vocab
        assert "und" in vocab
        assert len(vocab) == 3  # hallo, und, ja

    def test_load_vocab_missing_level(self, tmp_path):
        from audio2subtitle.filter import VocabFilter
        vocab = VocabFilter.load_vocab(str(tmp_path), ["Z99"])
        assert len(vocab) == 0

    def test_filter_keeps_advanced_segments(self):
        from audio2subtitle.filter import VocabFilter
        from audio2subtitle.models import SubtitleEntry

        mock_nlp = MagicMock()
        # "Quantenphysik" lemma not in basic vocab
        mock_token1 = MagicMock()
        mock_token1.is_punct = False
        mock_token1.is_stop = False
        mock_token1.pos_ = "NOUN"
        mock_token1.lemma_ = "Quantenphysik"
        mock_doc = [mock_token1]
        mock_nlp.return_value = mock_doc

        entry = SubtitleEntry(index=1, start_time=0.0, end_time=2.5, text="Quantenphysik ist komplex")
        result = VocabFilter.filter_segments([entry], mock_nlp, {"hallo", "und"})
        assert len(result) == 1

    def test_filter_removes_basic_segments(self):
        from audio2subtitle.filter import VocabFilter
        from audio2subtitle.models import SubtitleEntry

        mock_nlp = MagicMock()
        mock_token = MagicMock()
        mock_token.is_punct = False
        mock_token.is_stop = False
        mock_token.pos_ = "NOUN"
        mock_token.lemma_ = "hallo"
        mock_doc = [mock_token]
        mock_nlp.return_value = mock_doc

        entry = SubtitleEntry(index=1, start_time=0.0, end_time=2.5, text="hallo")
        result = VocabFilter.filter_segments([entry], mock_nlp, {"hallo", "und"})
        assert len(result) == 0

    def test_filter_skips_punct_stop_propn(self):
        from audio2subtitle.filter import VocabFilter
        from audio2subtitle.models import SubtitleEntry

        mock_nlp = MagicMock()
        # All tokens are punctuation/stop/proper noun → no lemmas → segment dropped
        mock_token = MagicMock()
        mock_token.is_punct = True
        mock_token.is_stop = False
        mock_token.pos_ = "PUNCT"
        mock_token.lemma_ = "."
        mock_nlp.return_value = [mock_token]

        entry = SubtitleEntry(index=1, start_time=0.0, end_time=2.5, text=".")
        result = VocabFilter.filter_segments([entry], mock_nlp, {"hallo"})
        assert len(result) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/audio2subtitle/tests/test_filter.py -v`
Expected: FAIL — `audio2subtitle.filter` module doesn't exist

- [ ] **Step 3: Implement filter module**

```python
# packages/audio2subtitle/src/audio2subtitle/filter.py

from pathlib import Path

import pandas as pd

from .models import SubtitleEntry


class VocabFilter:
    """Filter subtitle segments by vocabulary level using spaCy lemmatization."""

    @staticmethod
    def load_vocab(vocab_dir: str, levels: list[str]) -> set[str]:
        """Load vocabulary CSVs from directory. Each CSV: one word per row, no header."""
        vocab: set[str] = set()
        for level in levels:
            path = Path(vocab_dir) / f"{level}_vokabeln.csv"
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path, header=None)
                words = df.iloc[:, 0].astype(str).str.strip().str.lower().tolist()
                vocab.update(words)
            except Exception:
                continue
        return vocab

    @staticmethod
    def filter_segments(
        entries: list[SubtitleEntry],
        nlp: object,
        vocab_set: set[str],
    ) -> list[SubtitleEntry]:
        """Keep segments that contain at least one word NOT in vocab_set."""
        filtered: list[SubtitleEntry] = []
        for entry in entries:
            doc = nlp(entry.text)  # type: ignore[operator]
            lemmas: list[str] = []
            for token in doc:
                if token.is_punct or token.is_stop or token.pos_ in ("PROPN", "NUM", "INTJ", "X", "SPACE"):
                    continue
                lemma = token.lemma_.lower().strip()
                if lemma:
                    lemmas.append(lemma)
            if lemmas and not all(l in vocab_set for l in lemmas):
                filtered.append(entry)
        return filtered
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/audio2subtitle/tests/test_filter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/audio2subtitle/src/audio2subtitle/filter.py packages/audio2subtitle/tests/test_filter.py
git commit -m "feat(audio2subtitle): add VocabFilter for vocabulary-level segment filtering"
```

---

### Task 3: Translation Module

**Files:**
- Create: `packages/audio2subtitle/src/audio2subtitle/translator.py`
- Create: `packages/audio2subtitle/tests/test_translator.py`

- [ ] **Step 1: Write the failing tests**

```python
# packages/audio2subtitle/tests/test_translator.py

import pytest
import sys
from unittest.mock import MagicMock


class TestMarianTranslator:
    def test_translate_batch_calls_model(self, mocker):
        mock_ct2 = MagicMock()
        mock_transformers = MagicMock()
        mocker.patch.dict(sys.modules, {
            "ctranslate2": mock_ct2,
            "transformers": mock_transformers,
        })

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_ct2.Translator.return_value = mock_model

        # Mock tokenizer encode -> tokens
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.convert_ids_to_tokens.return_value = ["▁Hallo", "▁Welt"]

        # Mock translate_batch result
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
        # Should have called translate_batch twice: 16 + 19
        assert mock_model.translate_batch.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/audio2subtitle/tests/test_translator.py -v`
Expected: FAIL — `audio2subtitle.translator` module doesn't exist

- [ ] **Step 3: Implement translator module**

```python
# packages/audio2subtitle/src/audio2subtitle/translator.py

import ctranslate2
import transformers


class MarianTranslator:
    """Batch translator using CTranslate2-optimized MarianMT models. Fast CPU inference."""

    def __init__(self, model_name: str, ct2_model_path: str):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.translator = ctranslate2.Translator(ct2_model_path)

    def translate_batch(self, texts: list[str], batch_size: int = 32) -> list[str]:
        """Translate a list of texts in batches. Returns list of translated strings."""
        if not texts:
            return []
        translations: list[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            source_tokens = [
                self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
                for text in batch
            ]
            results = self.translator.translate_batch(source_tokens)
            for result in results:
                target_tokens = result.hypotheses[0]
                target_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
                translations.append(self.tokenizer.decode(target_ids, skip_special_tokens=True))
        return translations
```

**Note:** The CTranslate2 model must be pre-converted:
```bash
ct2-transformers-converter --model Helsinki-NLP/opus-mt-tc-big-de-es --output_dir /path/to/ct2-model
```
The `ct2_model_path` parameter points to this converted directory. The `model_name` parameter is for the HuggingFace tokenizer (which doesn't need conversion).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/audio2subtitle/tests/test_translator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/audio2subtitle/src/audio2subtitle/translator.py packages/audio2subtitle/tests/test_translator.py
git commit -m "feat(audio2subtitle): add MarianTranslator for DE→ES batch translation"
```

---

### Task 4: Wire Filter + Translate into Provider

**Files:**
- Modify: `packages/audio2subtitle/src/audio2subtitle/providers/mlx.py`
- Modify: `packages/audio2subtitle/tests/test_audio2subtitle.py`

- [ ] **Step 1: Write the failing tests for provider with filter+translate**

```python
# Add to packages/audio2subtitle/tests/test_audio2subtitle.py

def test_mlx_provider_with_translation(tmp_path, mocker):
    _mock_whisper = MagicMock()
    mocker.patch.dict(sys.modules, {"mlx_whisper": _mock_whisper})
    _mock_whisper.transcribe.return_value = {
        "language": "de",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Hallo Welt."},
        ],
    }

    mock_translator = MagicMock()
    mock_translator.translate_batch.return_value = ["Hola Mundo."]
    mocker.patch("audio2subtitle.providers.mlx.MarianTranslator", return_value=mock_translator)

    from audio2subtitle.providers.mlx import MLXWhisperProvider

    provider = MLXWhisperProvider()
    request = Audio2SubtitleRequest(audio_path="/tmp/audio.wav", language="de", translate_to="es", ct2_model_path="/tmp/ct2-model")
    out_file = tmp_path / "out.srt"
    response = provider.generate(request, str(out_file))

    assert len(response.entries) == 1
    assert response.entries[0].text == "Hallo Welt."
    assert response.entries[0].translated_text == "Hola Mundo."
    content = out_file.read_text()
    assert "Hola Mundo." in content


def test_mlx_provider_with_vocab_filter(tmp_path, mocker):
    _mock_whisper = MagicMock()
    mocker.patch.dict(sys.modules, {"mlx_whisper": _mock_whisper})
    _mock_whisper.transcribe.return_value = {
        "language": "de",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Hallo."},
            {"start": 2.5, "end": 5.0, "text": "Quantenphysik ist komplex."},
        ],
    }

    mock_filter_mod = MagicMock()
    mocker.patch("audio2subtitle.providers.mlx.VocabFilter", mock_filter_mod)

    # First entry filtered out, second kept
    from audio2subtitle.models import SubtitleEntry
    mock_filter_mod.filter_segments.return_value = [
        SubtitleEntry(index=1, start_time=2.5, end_time=5.0, text="Quantenphysik ist komplex.")
    ]
    mock_filter_mod.load_vocab.return_value = {"hallo"}

    mock_nlp = MagicMock()
    mocker.patch("audio2subtitle.providers.mlx.spacy")
    mocker.patch("audio2subtitle.providers.mlx.spacy.load", return_value=mock_nlp)

    from audio2subtitle.providers.mlx import MLXWhisperProvider

    provider = MLXWhisperProvider()
    request = Audio2SubtitleRequest(
        audio_path="/tmp/audio.wav", language="de", vocab_filter_path="/data/vocab"
    )
    out_file = tmp_path / "out.srt"
    response = provider.generate(request, str(out_file))

    assert len(response.entries) == 1
    assert "Quantenphysik" in response.entries[0].text


def test_mlx_provider_filter_then_translate(tmp_path, mocker):
    _mock_whisper = MagicMock()
    mocker.patch.dict(sys.modules, {"mlx_whisper": _mock_whisper})
    _mock_whisper.transcribe.return_value = {
        "language": "de",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Quantenphysik ist komplex."},
        ],
    }

    mock_filter_mod = MagicMock()
    mocker.patch("audio2subtitle.providers.mlx.VocabFilter", mock_filter_mod)
    from audio2subtitle.models import SubtitleEntry
    mock_filter_mod.filter_segments.return_value = [
        SubtitleEntry(index=1, start_time=0.0, end_time=2.5, text="Quantenphysik ist komplex.")
    ]
    mock_filter_mod.load_vocab.return_value = {"hallo"}

    mocker.patch("audio2subtitle.providers.mlx.spacy")
    mocker.patch("audio2subtitle.providers.mlx.spacy.load", return_value=MagicMock())

    mock_translator = MagicMock()
    mock_translator.translate_batch.return_value = ["La física cuántica es compleja."]
    mocker.patch("audio2subtitle.providers.mlx.MarianTranslator", return_value=mock_translator)

    from audio2subtitle.providers.mlx import MLXWhisperProvider

    provider = MLXWhisperProvider()
    request = Audio2SubtitleRequest(
        audio_path="/tmp/audio.wav",
        language="de",
        translate_to="es",
        ct2_model_path="/tmp/ct2-model",
        vocab_filter_path="/data/vocab",
    )
    out_file = tmp_path / "out.srt"
    response = provider.generate(request, str(out_file))

    assert len(response.entries) == 1
    assert response.entries[0].translated_text == "La física cuántica es compleja."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/audio2subtitle/tests/test_audio2subtitle.py -k "translation or vocab_filter or filter_then_translate" -v`
Expected: FAIL — provider doesn't call filter or translator

- [ ] **Step 3: Implement provider changes**

```python
# packages/audio2subtitle/src/audio2subtitle/providers/mlx.py

from pathlib import Path

from aiservices_core.providers import BaseProvider

from ..models import Audio2SubtitleRequest, Audio2SubtitleResponse, SubtitleEntry


def _format_timestamp(seconds: float, fmt: str = "srt") -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    if fmt == "vtt":
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class MLXWhisperProvider(BaseProvider):
    """Audio-to-subtitle provider using mlx-whisper (Apple Silicon)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(
        self, request: Audio2SubtitleRequest, output_path: str | None = None
    ) -> Audio2SubtitleResponse:
        try:
            import mlx_whisper

            result = mlx_whisper.transcribe(
                request.audio_path,
                path_or_hf_repo=request.model_name,
                language=request.language,
                word_timestamps=request.word_timestamps,
            )
        except Exception as e:
            raise RuntimeError(f"Subtitle generation failed: {e}") from e

        segments = result.get("segments", [])
        entries: list[SubtitleEntry] = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            entries.append(
                SubtitleEntry(
                    index=len(entries) + 1,
                    start_time=seg.get("start", 0.0),
                    end_time=seg.get("end", 0.0),
                    text=text,
                )
            )

        # Optional: vocabulary filtering
        if request.vocab_filter_path:
            from ..filter import VocabFilter
            import spacy

            vocab_set = VocabFilter.load_vocab(request.vocab_filter_path, request.vocab_levels)
            nlp = spacy.load("de_core_news_lg", disable=["parser", "ner"])
            entries = VocabFilter.filter_segments(entries, nlp, vocab_set)
            # Re-index after filtering
            for i, entry in enumerate(entries, 1):
                entry.index = i

        # Optional: translation
        if request.translate_to and entries:
            from ..translator import MarianTranslator

            translator = MarianTranslator(request.translation_model, request.ct2_model_path)
            texts = [e.text for e in entries]
            translated = translator.translate_batch(texts)
            for entry, trans in zip(entries, translated):
                entry.translated_text = trans

        # Write output
        fmt = request.output_format
        if fmt not in {"srt", "vtt"}:
            raise ValueError(f"Unsupported output format: {fmt}")
        output = output_path or f"output.{fmt}"
        Path(output).parent.mkdir(parents=True, exist_ok=True)

        if fmt == "vtt":
            content = "WEBVTT\n\n"
            for entry in entries:
                start = _format_timestamp(entry.start_time, "vtt")
                end = _format_timestamp(entry.end_time, "vtt")
                text = entry.translated_text or entry.text
                content += f"{start} --> {end}\n{text}\n\n"
        else:
            content = ""
            for entry in entries:
                start = _format_timestamp(entry.start_time, "srt")
                end = _format_timestamp(entry.end_time, "srt")
                text = entry.translated_text or entry.text
                content += f"{entry.index}\n{start} --> {end}\n{text}\n\n"

        with open(output, "w", encoding="utf-8") as f:
            f.write(content)

        return Audio2SubtitleResponse(
            output_path=output,
            entries=entries,
            language=str(result.get("language")) if result.get("language") else None,
            metadata={
                "provider": "mlx-whisper",
                "model": request.model_name,
                "format": fmt,
                "total_entries": len(entries),
                "word_timestamps": request.word_timestamps,
                "filtered": request.vocab_filter_path is not None,
                "translated": request.translate_to is not None,
            },
        )
```

- [ ] **Step 4: Run all tests**

Run: `uv run pytest packages/audio2subtitle/tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add packages/audio2subtitle/src/audio2subtitle/providers/mlx.py packages/audio2subtitle/tests/test_audio2subtitle.py
git commit -m "feat(audio2subtitle): wire vocab filtering and translation into MLX provider"
```

---

### Task 5: Update CLI

**Files:**
- Modify: `packages/audio2subtitle/src/audio2subtitle/cli.py`
- Modify: `packages/audio2subtitle/tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
# Add to packages/audio2subtitle/tests/test_cli.py

def test_transcribe_with_translate_to(tmp_path, mocker):
    mock_registry = mocker.patch("audio2subtitle.cli.registry")
    mock_provider = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.output_path = str(tmp_path / "out.srt")
    mock_response.entries = []
    mock_response.language = "de"
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.srt"
    result = runner.invoke(
        app,
        ["--audio", "/tmp/audio.wav", "--output", str(out), "--language", "de", "--translate-to", "es"],
    )
    assert result.exit_code == 0


def test_transcribe_with_vocab_filter(tmp_path, mocker):
    mock_registry = mocker.patch("audio2subtitle.cli.registry")
    mock_provider = mocker.MagicMock()
    mock_response = mocker.MagicMock()
    mock_response.output_path = str(tmp_path / "out.srt")
    mock_response.entries = []
    mock_response.language = "de"
    mock_provider.generate.return_value = mock_response
    mock_registry.get.return_value = mock_provider

    out = tmp_path / "out.srt"
    result = runner.invoke(
        app,
        ["--audio", "/tmp/audio.wav", "--output", str(out), "--vocab-filter", "/data/vocab"],
    )
    assert result.exit_code == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/audio2subtitle/tests/test_cli.py -k "translate_to or vocab_filter" -v`
Expected: FAIL — CLI options don't exist

- [ ] **Step 3: Implement CLI changes**

```python
# packages/audio2subtitle/src/audio2subtitle/cli.py

import time
from typing import Literal, cast

import typer
from aiservices_core.cli import device_option, verbose_option
from aiservices_core.logging import create_progress_bar, get_logger

from .models import Audio2SubtitleRequest
from .providers import registry

app = typer.Typer(help="Audio to subtitle (SRT/VTT) transcription pipeline")
logger = get_logger(__name__)


@app.command()
def transcribe(
    audio: str = typer.Option(..., "--audio", "-a", help="Path to input audio file"),
    output: str = typer.Option(..., "--output", "-o", help="Path to output subtitle file"),
    format: str = typer.Option("srt", "--format", "-f", help="Output format: srt or vtt"),
    language: str = typer.Option(None, "--language", "-l", help="Language code (e.g. 'de')"),
    model: str = typer.Option(
        "mlx-community/whisper-large-v3-turbo",
        "--model",
        help="Whisper model name",
    ),
    no_word_timestamps: bool = typer.Option(
        False, "--no-word-timestamps", help="Disable word-level timestamps"
    ),
    translate_to: str = typer.Option(
        None, "--translate-to", help="Target language for translation (e.g. 'es')"
    ),
    translation_model: str = typer.Option(
        "Helsinki-NLP/opus-mt-tc-big-de-es", "--translation-model", help="Translation model name (for tokenizer)"
    ),
    ct2_model_path: str = typer.Option(
        "", "--ct2-model-path", help="Path to CTranslate2-converted model directory"
    ),
    vocab_filter: str = typer.Option(
        None, "--vocab-filter", help="Path to vocab CSV directory for filtering"
    ),
    vocab_levels: str = typer.Option(
        "A1,A2,B1", "--vocab-levels", help="Comma-separated vocab levels"
    ),
    provider_name: str = typer.Option("audio2subtitle.mlx", "--provider", help="Provider name"),
    verbose: bool = verbose_option,
    device: str = device_option,
):
    """Transcribe audio to subtitle file (SRT/VTT), with optional filtering and translation."""
    levels = [l.strip() for l in vocab_levels.split(",")]

    request = Audio2SubtitleRequest(
        audio_path=audio,
        language=language,
        output_format=cast(Literal["srt", "vtt"], format),
        model_name=model,
        word_timestamps=not no_word_timestamps,
        translate_to=translate_to,
        translation_model=translation_model,
        ct2_model_path=ct2_model_path,
        vocab_filter_path=vocab_filter,
        vocab_levels=levels,
    )

    logger.info(f"Using provider: {provider_name}")

    try:
        with create_progress_bar() as progress:
            task_id = progress.add_task("[cyan]Initializing provider...", total=None)
            provider = registry.get(provider_name, device=device)

            progress.update(task_id, description="[green]Transcribing audio to subtitles...")
            start_time = time.time()
            response = provider.generate(request, output)
            elapsed = time.time() - start_time
            progress.update(task_id, description=f"[bold green]Done in {elapsed:.2f}s!")

        typer.echo(f"\nSubtitle file saved to: {response.output_path}")
        typer.echo(f"Detected language: {response.language}")
        typer.echo(f"Entries: {len(response.entries)}")
    except Exception as e:
        logger.error(f"Subtitle generation failed: {str(e)}")
        typer.secho(f"\nError: {str(e)}", fg=typer.colors.RED, bold=True, err=True)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run all tests**

Run: `uv run pytest packages/audio2subtitle/tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add packages/audio2subtitle/src/audio2subtitle/cli.py packages/audio2subtitle/tests/test_cli.py
git commit -m "feat(audio2subtitle): add --translate-to, --vocab-filter, --vocab-levels CLI options"
```

---

### Task 6: Update Dependencies

**Files:**
- Modify: `packages/audio2subtitle/pyproject.toml`

- [ ] **Step 1: Add optional dependencies**

```toml
# packages/audio2subtitle/pyproject.toml

[project]
name = "audio2subtitle"
version = "0.1.0"
description = "Audio to subtitle (SRT/VTT) transcription pipeline with optional filtering and translation"
requires-python = ">=3.10"
dependencies = [
    "aiservices-core",
    "mlx-whisper>=0.4.3",
    "pydantic>=2.0.0",
    "typer>=0.12.0",
]

[project.optional-dependencies]
filter = [
    "spacy>=3.0.0",
    "pandas>=2.0.0",
]
translate = [
    "ctranslate2>=4.0.0",
    "transformers>=4.30.0",
    "sentencepiece>=0.1.99",
]
all = [
    "audio2subtitle[filter,translate]",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project.scripts]
audio2subtitle = "audio2subtitle.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/audio2subtitle"]

[tool.uv.sources]
aiservices-core = { workspace = true }
```

- [ ] **Step 2: Sync dependencies**

Run: `uv sync --all-packages`
Expected: SUCCESS

- [ ] **Step 3: Run all tests**

Run: `uv run pytest packages/audio2subtitle/tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Lint and typecheck**

Run: `uvx ruff check packages/audio2subtitle/ && uvx pyright packages/audio2subtitle/`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add packages/audio2subtitle/pyproject.toml
git commit -m "feat(audio2subtitle): add optional filter/translate dependencies"
```

---

## Final Usage

```bash
# Install with all features
uv pip install "audio2subtitle[all]"

# Transcribe only (existing behavior, unchanged)
audio2subtitle transcribe --audio lecture.m4a --output lecture.srt

# Full Colab pipeline: transcribe + filter + translate
audio2subtitle transcribe \
  --audio lecture.m4a \
  --output lecture.srt \
  --language de \
  --translate-to es \
  --ct2-model-path ./ct2-opus-mt-de-es \
  --vocab-filter ./IdeaProjects/src/backend/data/ \
  --vocab-levels A1,A2,B1

# Translate only (no filtering)
audio2subtitle transcribe \
  --audio lecture.m4a \
  --output lecture.srt \
  --language de \
  --translate-to es \
  --ct2-model-path ./ct2-opus-mt-de-es

# Filter only (no translation)
audio2subtitle transcribe \
  --audio lecture.m4a \
  --output lecture.srt \
  --language de \
  --vocab-filter ./IdeaProjects/src/backend/data/
```
