import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from studio.models import Episode, Scene, Shot, Dialogue
from studio.orchestrator.showrunner import Showrunner


@pytest.fixture
def dummy_episode():
    return Episode(
        title="Test Episode",
        cast={"char_1": {"voice_id": "v1"}},
        scenes=[
            Scene(
                scene_id="1",
                title="Scene 1",
                location="Kitchen",
                description="A test scene",
                shots=[
                    Shot(
                        shot_id="1_1",
                        description="A test shot",
                        dialogue=[
                            Dialogue(speaker="char_1", text="Hello world")
                        ]
                    )
                ]
            )
        ]
    )


@patch("studio.orchestrator.showrunner.t2v_registry")
@patch("studio.orchestrator.showrunner.tts_registry")
@patch("ffmpeg.run")
def test_showrunner_orchestration(mock_ffmpeg, mock_tts_reg, mock_t2v_reg, dummy_episode, tmp_path):
    # Setup mocks
    mock_tts = MagicMock()
    mock_tts_reg.get.return_value = mock_tts
    
    mock_t2v = MagicMock()
    mock_t2v_reg.get.return_value = mock_t2v
    
    showrunner = Showrunner(
        output_dir=tmp_path,
        tts_provider="mock_tts",
        t2v_provider="mock_t2v"
    )
    
    # We need to mock Path.exists for audio combining logic
    with patch("studio.orchestrator.showrunner.Path.exists", return_value=True):
        final_path = showrunner.render_episode(dummy_episode)
    
    # Verify final path
    assert final_path == tmp_path / "Test_Episode.mp4"
    
    # Verify provider calls
    mock_tts.generate.assert_called()
    mock_t2v.generate.assert_called()
    
    # Verify ffmpeg calls
    assert mock_ffmpeg.call_count >= 3 # 1 shot merge, 1 scene concat, 1 episode concat
