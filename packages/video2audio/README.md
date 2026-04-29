# video2audio

Video to audio extraction and generation pipeline (M5).

## CLI Usage

```bash
video2audio extract --video input.mp4 --output audio.wav
video2audio extract --video input.mp4 --output audio.mp3 --format mp3
video2audio extract --video input.mp4 --output audio.wav --sample-rate 48000 --mono
```

## Python API

```python
from video2audio.models import Video2AudioRequest
from video2audio.providers import registry

request = Video2AudioRequest(video_path="input.mp4", output_format="wav")
provider = registry.get("video2audio.ffmpeg")
response = provider.generate(request, "output.wav")
print(response.output_path)
```

## Requirements

- FFmpeg installed and available on PATH
