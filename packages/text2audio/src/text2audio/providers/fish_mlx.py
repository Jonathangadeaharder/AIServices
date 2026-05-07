from aiservices_core.providers import BaseProvider

from ..models import Text2AudioRequest, Text2AudioResponse


class FishMLXProvider(BaseProvider):
    """Text-to-audio provider using Fish S2 Pro via mlx-audio (Apple Silicon).

    This provider is a placeholder for future implementation.
    The underlying model (Fish S2 Pro MLX) is not yet available for
    text-to-audio generation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(
        self, request: Text2AudioRequest, output_path: str | None = None
    ) -> Text2AudioResponse:
        raise NotImplementedError(
            "Fish S2 Pro MLX text-to-audio provider is not yet implemented. "
            "This is a placeholder for future development."
        )
