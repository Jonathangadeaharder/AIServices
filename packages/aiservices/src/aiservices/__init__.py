"""AIServices — unified AI service abstraction layer.

Callers import from aiservices, never from concrete providers directly.
This allows swapping backends without caller changes.

Supported models:
- Whisper: Speech-to-text via mlx-whisper (mlx-community/whisper-large-v3-turbo)
- Fish Speech: Text-to-speech via mlx-audio (fishaudio/fish-speech-1.5)
- Flux 2: Image generation + LoRA training via mflux (mlx-community/FLUX.2-klein-9B)
- LTX 2.3: Video generation via ltx-pipelines-mlx (dgrauet/ltx-2.3-mlx-q8)
  Including keyframe interpolation (start+end frame)
"""

from aiservices.generate import (
    AudioClip,
    AudioGenerator,
    GenConfig,
    ImageFrame,
    ImageGenerator,
    LoRAAdapter,
    VideoGenerator,
    VideoMode,
    generate_audio,
    generate_image,
    generate_image2image,
    generate_text2image,
    generate_video,
    _text2image,
    _image2image,
)
from aiservices.transcribe import (
    BaseTranscriber,
    FasterWhisperTranscriber,
    MLXWhisperTranscriber,
    Segment,
    TranscriptionResult,
    create_transcriber,
    transcribe,
    transcribe_to_srt,
)
from aiservices.translate import (
    BaseTranslator,
    NLLBTranslator,
    OpenAITranslator,
    Translation,
    create_translator,
    translate_text,
)

__all__ = [
    # Image
    "GenConfig",
    "ImageFrame",
    "ImageGenerator",
    "generate_image",
    "generate_image2image",
    "generate_text2image",
    "_text2image",
    "_image2image",
    # Audio
    "AudioClip",
    "AudioGenerator",
    "generate_audio",
    # Video
    "LoRAAdapter",
    "VideoGenerator",
    "VideoMode",
    "generate_video",
    # Transcription
    "BaseTranscriber",
    "FasterWhisperTranscriber",
    "MLXWhisperTranscriber",
    "Segment",
    "TranscriptionResult",
    "create_transcriber",
    "transcribe",
    "transcribe_to_srt",
    # Translation
    "BaseTranslator",
    "NLLBTranslator",
    "OpenAITranslator",
    "Translation",
    "create_translator",
    "translate_text",
]
