import io
import sys

import pysrt


def read_srt(input_path: str | None = None) -> pysrt.SubRipFile:
    """Read SRT from file path or stdin."""
    if input_path:
        return pysrt.open(input_path, encoding="utf-8")
    data = sys.stdin.read()
    return pysrt.SubRipFile.from_string(data)


def write_srt(subs: pysrt.SubRipFile, output_path: str | None = None) -> None:
    """Write SRT to file path or stdout."""
    buf = io.StringIO()
    subs.write(buf)
    content = buf.getvalue()
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        sys.stdout.write(content)
