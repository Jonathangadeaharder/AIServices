# Changelog — text2audio

All notable changes to this module will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-05-06

### Added

- Text-to-audio generation module (music, SFX, ambient, speech)
- Placeholder provider registry — no providers registered yet
- CLI: `text2audio generate --prompt <text> --output <wav>`
- Python API: `text2audio.models.Text2AudioRequest` + provider pattern
- Audio categories: music, sfx, ambient, speech
- Configurable duration, output format, seed
- Pydantic request/response models
- E2E tests
