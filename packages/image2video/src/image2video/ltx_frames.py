"""LTX-2 temporal constraints for image2video generation."""


def normalize_ltx_frame_count(seconds: float, fps: int) -> int:
    """Return a valid LTX frame count (8k + 1) for the requested duration.

    LTX VAE uses 8x temporal compression; frame counts must satisfy 8k+1 or
    decoded video corrupts after the conditioned first frame.
    """
    if fps < 1:
        raise ValueError(f"fps must be >= 1, got {fps}")
    if seconds <= 0:
        raise ValueError(f"seconds must be > 0, got {seconds}")

    target = max(9, int(round(seconds * fps)))
    k = max(1, (target - 1 + 7) // 8)
    return k * 8 + 1
