import os
import base64
from io import BytesIO
from typing import List
import cv2
from PIL import Image
from openai import OpenAI
import structlog

logger = structlog.get_logger()

class MultimodalJudge:
    """
    A clean abstraction for a local multimodal judge (Nemotron Omni).
    Handles media processing and local AI orchestration.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "local"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = os.getenv("JUDGE_MODEL", "nemotron-omni")

    def judge_image(self, image_path: str, prompt: str) -> str:
        """
        Judge a single image file.
        """
        if not os.path.exists(image_path):
            logger.error("image_not_found", path=image_path)
            raise FileNotFoundError(f"Image file not found: {image_path}")

        logger.info("judging_image", path=image_path)
        with Image.open(image_path) as im:
            b64_image = self._encode_image(im)

        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=500
        )
        
        return response.choices[0].message.content

    def judge_video(self, video_path: str, prompt: str, frame_count: int = 5) -> str:
        """
        Judge a video file by extracting key frames and sending them to the multimodal model.
        """
        if not os.path.exists(video_path):
            logger.error("video_not_found", path=video_path)
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if frame_count < 1:
            logger.warning("invalid_frame_count", frame_count=frame_count)
            frame_count = 1

        logger.info("judging_video", path=video_path, frames=frame_count)
        frames = self._extract_frames(video_path, frame_count)
        
        if not frames:
            logger.warning("no_frames_extracted", path=video_path)

        content = [{"type": "text", "text": prompt}]
        for frame in frames:
            b64_image = self._encode_image(frame)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
            })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=500
        )
        
        return response.choices[0].message.content

    def _extract_frames(self, video_path: str, count: int) -> List[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("failed_to_open_video", path=video_path)
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logger.warning("empty_video", path=video_path)
            cap.release()
            return []

        step = max(1, total_frames // count)
        
        frames = []
        for i in range(count):
            frame_idx = i * step
            if frame_idx >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if success:
                # Convert BGR (OpenCV) to RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        return frames

    def _encode_image(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
