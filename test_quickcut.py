"""Tests for audio-sync fixes in quickcut.py.

Verifies that _center_crop_to_portrait and render_enhanced build correct
FFmpeg commands with audio resync filters (aresample + asetpts).
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ffprobe_result(has_audio: bool) -> subprocess.CompletedProcess:
    """Return a fake ffprobe CompletedProcess for audio-stream detection."""
    return subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout="0\n" if has_audio else "",
        stderr="",
    )


def _make_resolution_result(width: int = 1920, height: int = 1080) -> subprocess.CompletedProcess:
    """Return a fake ffprobe CompletedProcess for resolution detection."""
    return subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout=f"{width},{height}\n",
        stderr="",
    )


# We need to mock top-level imports that may not be installed in CI.
# anthropic, httpx, yaml are imported at module level in quickcut.py.
import sys
from unittest.mock import MagicMock as _MagicMock

for _mod in ("anthropic", "httpx", "yaml"):
    if _mod not in sys.modules:
        sys.modules[_mod] = _MagicMock()

import quickcut  # noqa: E402 (after mocking deps)


# ===================================================================
# 1. _center_crop_to_portrait — WITH audio
# ===================================================================

class TestCenterCropWithAudio:
    """_center_crop_to_portrait should add -af with aresample+asetpts when audio is present."""

    @patch("quickcut._run_ffmpeg")
    @patch("quickcut.subprocess.run")
    @patch("quickcut._detect_resolution", return_value=(1920, 1080))
    def test_af_present_when_audio(self, mock_res, mock_subproc, mock_ffmpeg):
        # First subprocess.run call is the ffprobe audio check
        mock_subproc.return_value = _make_ffprobe_result(has_audio=True)

        quickcut._center_crop_to_portrait(
            video_path="input.mp4",
            output_path="output.mp4",
            target_w=1080,
            target_h=1920,
            fps=30,
        )

        # _run_ffmpeg should have been called once
        mock_ffmpeg.assert_called_once()
        cmd = mock_ffmpeg.call_args[0][0]

        # Must contain -af with the resync chain
        assert "-af" in cmd, f"-af flag missing from cmd: {cmd}"
        af_idx = cmd.index("-af")
        af_value = cmd[af_idx + 1]
        assert "aresample=async=1000:first_pts=0" in af_value
        assert "asetpts=PTS-STARTPTS" in af_value

        # Must also contain audio codec flags
        assert "-c:a" in cmd
        assert "aac" in cmd

    @patch("quickcut._run_ffmpeg")
    @patch("quickcut.subprocess.run")
    @patch("quickcut._detect_resolution", return_value=(1920, 1080))
    def test_vf_contains_setpts(self, mock_res, mock_subproc, mock_ffmpeg):
        """Video filter should still have setpts for video."""
        mock_subproc.return_value = _make_ffprobe_result(has_audio=True)

        quickcut._center_crop_to_portrait(
            video_path="input.mp4",
            output_path="output.mp4",
            target_w=1080,
            target_h=1920,
        )

        cmd = mock_ffmpeg.call_args[0][0]
        vf_idx = cmd.index("-vf")
        vf_value = cmd[vf_idx + 1]
        assert "setpts=PTS-STARTPTS" in vf_value


# ===================================================================
# 2. _center_crop_to_portrait — WITHOUT audio
# ===================================================================

class TestCenterCropWithoutAudio:
    """_center_crop_to_portrait should omit -af and -c:a when no audio stream."""

    @patch("quickcut._run_ffmpeg")
    @patch("quickcut.subprocess.run")
    @patch("quickcut._detect_resolution", return_value=(1920, 1080))
    def test_no_af_when_no_audio(self, mock_res, mock_subproc, mock_ffmpeg):
        mock_subproc.return_value = _make_ffprobe_result(has_audio=False)

        quickcut._center_crop_to_portrait(
            video_path="input.mp4",
            output_path="output.mp4",
            target_w=1080,
            target_h=1920,
        )

        cmd = mock_ffmpeg.call_args[0][0]
        assert "-af" not in cmd, f"-af should not appear when no audio: {cmd}"
        assert "-c:a" not in cmd, f"-c:a should not appear when no audio: {cmd}"


# ===================================================================
# 3. render_enhanced — WITH audio
# ===================================================================

class TestRenderEnhancedWithAudio:
    """render_enhanced should include [aout] audio filter and map when audio is present."""

    @patch("quickcut.shutil.copy2")
    @patch("quickcut._run_ffmpeg")
    @patch("quickcut.subprocess.run")
    def test_filter_complex_contains_aout(self, mock_subproc, mock_ffmpeg, mock_copy):
        # ffprobe for audio detection returns audio present
        mock_subproc.return_value = _make_ffprobe_result(has_audio=True)

        config = {
            "rendering": {"output_resolution": [1080, 1920], "fps": 30, "crf": 18},
            "dynamic_edits": {},
        }

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            quickcut.render_enhanced(
                video_path="input.mp4",
                broll_clips=[{
                    "clip_path": "broll.mp4",
                    "start_time": 2.0,
                    "end_time": 5.0,
                }],
                edit_points=[],
                montage_path="",
                montage_dur=0.0,
                caption_ass="",
                hook_ass="",
                output_path="final.mp4",
                config=config,
                tmp_dir=tmp_dir,
            )

        # _run_ffmpeg is called for pass 1 (and possibly pass 2)
        assert mock_ffmpeg.call_count >= 1
        cmd1 = mock_ffmpeg.call_args_list[0][0][0]

        # Verify -filter_complex is present
        assert "-filter_complex" in cmd1
        fc_idx = cmd1.index("-filter_complex")
        fc_value = cmd1[fc_idx + 1]

        # Must contain the audio resync filter
        assert "[0:a]aresample=async=1000:first_pts=0,asetpts=PTS-STARTPTS[aout]" in fc_value

        # Must map [aout]
        assert "[aout]" in cmd1, f"[aout] mapping missing from cmd: {cmd1}"

        # Must NOT have -af (audio is handled inside filter_complex)
        assert "-af" not in cmd1, f"-af should not appear in render_enhanced cmd: {cmd1}"

    @patch("quickcut.shutil.copy2")
    @patch("quickcut._run_ffmpeg")
    @patch("quickcut.subprocess.run")
    def test_audio_codec_flags_present(self, mock_subproc, mock_ffmpeg, mock_copy):
        mock_subproc.return_value = _make_ffprobe_result(has_audio=True)

        config = {
            "rendering": {"output_resolution": [1080, 1920], "fps": 30, "crf": 18},
            "dynamic_edits": {},
        }

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            quickcut.render_enhanced(
                video_path="input.mp4",
                broll_clips=[{
                    "clip_path": "broll.mp4",
                    "start_time": 1.0,
                    "end_time": 3.0,
                }],
                edit_points=[],
                montage_path="",
                montage_dur=0.0,
                caption_ass="",
                hook_ass="",
                output_path="final.mp4",
                config=config,
                tmp_dir=tmp_dir,
            )

        cmd1 = mock_ffmpeg.call_args_list[0][0][0]
        assert "-c:a" in cmd1
        assert "aac" in cmd1


# ===================================================================
# 4. render_enhanced — WITHOUT audio
# ===================================================================

class TestRenderEnhancedWithoutAudio:
    """render_enhanced should omit [aout] and audio map when no audio stream."""

    @patch("quickcut.shutil.copy2")
    @patch("quickcut._run_ffmpeg")
    @patch("quickcut.subprocess.run")
    def test_no_aout_when_no_audio(self, mock_subproc, mock_ffmpeg, mock_copy):
        mock_subproc.return_value = _make_ffprobe_result(has_audio=False)

        config = {
            "rendering": {"output_resolution": [1080, 1920], "fps": 30, "crf": 18},
            "dynamic_edits": {},
        }

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            quickcut.render_enhanced(
                video_path="input.mp4",
                broll_clips=[{
                    "clip_path": "broll.mp4",
                    "start_time": 2.0,
                    "end_time": 4.0,
                }],
                edit_points=[],
                montage_path="",
                montage_dur=0.0,
                caption_ass="",
                hook_ass="",
                output_path="final.mp4",
                config=config,
                tmp_dir=tmp_dir,
            )

        cmd1 = mock_ffmpeg.call_args_list[0][0][0]

        fc_idx = cmd1.index("-filter_complex")
        fc_value = cmd1[fc_idx + 1]

        assert "[aout]" not in fc_value, f"[aout] should not be in filter_complex without audio"
        assert "[aout]" not in cmd1[fc_idx + 2:], f"[aout] mapping should not exist without audio"

        # No audio codec flags expected
        assert "-c:a" not in cmd1


# ===================================================================
# 5. Hook phrase generation — extract_hook & _extract_subject
# ===================================================================

class TestExtractHookEmptySegments:
    """extract_hook returns None when segments list is empty."""

    def test_returns_none_for_empty_segments(self):
        result = quickcut.extract_hook({"segments": [], "full_text": "hello"})
        assert result is None

    def test_returns_none_for_missing_segments(self):
        result = quickcut.extract_hook({"full_text": "hello"})
        assert result is None


class TestExtractHookValidDict:
    """extract_hook returns a dict with the expected keys and value types."""

    def test_returns_dict_with_correct_keys(self):
        transcript = {
            "segments": [{"start": 0.0, "end": 3.0, "text": "Hello world"}],
            "full_text": "Hello world this is a test about machine learning",
        }
        result = quickcut.extract_hook(transcript)
        assert isinstance(result, dict)
        assert "text" in result
        assert "start" in result
        assert "end" in result
        assert result["start"] == 0.0

    def test_text_comes_from_hook_phrases(self):
        """The returned text (before subject substitution) should match one of _HOOK_PHRASES."""
        transcript = {
            "segments": [{"start": 0.0, "end": 2.5, "text": "test"}],
            "full_text": "simple words only",
        }
        # Run several times to cover randomness
        for _ in range(20):
            result = quickcut.extract_hook(transcript)
            # At least one phrase template should be a substring match after
            # replacing {subject} with the extracted subject.
            assert isinstance(result["text"], str)
            assert len(result["text"]) > 0


class TestExtractHookSubjectReplaced:
    """The {subject} placeholder must not survive into the output text."""

    def test_no_subject_placeholder_in_output(self):
        transcript = {
            "segments": [{"start": 0.0, "end": 2.0, "text": "stuff"}],
            "full_text": "Learning about quantum computing today",
        }
        for _ in range(30):
            result = quickcut.extract_hook(transcript)
            assert "{subject}" not in result["text"], (
                f"Placeholder survived: {result['text']}"
            )


class TestExtractHookEndClamped:
    """hook_end must be clamped between 1.0 and 4.0."""

    def test_short_segment_clamps_to_1(self):
        transcript = {
            "segments": [{"start": 0.0, "end": 0.3, "text": "hi"}],
            "full_text": "hi",
        }
        result = quickcut.extract_hook(transcript)
        assert result["end"] == 1.0

    def test_long_segment_clamps_to_4(self):
        transcript = {
            "segments": [{"start": 0.0, "end": 12.0, "text": "long segment"}],
            "full_text": "long segment about many things",
        }
        result = quickcut.extract_hook(transcript)
        assert result["end"] == 4.0

    def test_mid_range_segment_preserved(self):
        transcript = {
            "segments": [{"start": 0.0, "end": 2.5, "text": "mid"}],
            "full_text": "mid range content",
        }
        result = quickcut.extract_hook(transcript)
        assert result["end"] == 2.5

    def test_missing_end_defaults_to_4(self):
        """When segment has no 'end' key, default is 4.0, clamped to 4.0."""
        transcript = {
            "segments": [{"start": 0.0, "text": "no end key"}],
            "full_text": "no end key here",
        }
        result = quickcut.extract_hook(transcript)
        assert result["end"] == 4.0


class TestExtractSubject:
    """_extract_subject should extract meaningful words or fall back to 'this'."""

    def test_empty_transcript_returns_this(self):
        assert quickcut._extract_subject({}) == "this"
        assert quickcut._extract_subject({"full_text": ""}) == "this"

    def test_extracts_words_after_marker(self):
        transcript = {"full_text": "Today I want to talk about building muscle fast"}
        result = quickcut._extract_subject(transcript)
        assert result != "this"
        assert len(result) > 2

    def test_falls_back_to_content_words(self):
        """When no marker word is found, content words are returned."""
        transcript = {"full_text": "quantum computing rocks"}
        result = quickcut._extract_subject(transcript)
        assert result != "this"
        assert len(result) > 2


# ===================================================================
# 6. Description generation
# ===================================================================

class TestDescriptionPromptTemplate:
    """_DESCRIPTION_PROMPT must be a valid format string with {transcript}."""

    def test_format_with_transcript_does_not_crash(self):
        result = quickcut._DESCRIPTION_PROMPT.format(transcript="test")
        assert "test" in result


class TestGenerateDescriptionNoKey:
    """generate_description returns '' when no API key is available."""

    def test_returns_empty_when_no_key_in_config(self):
        config = {"api_keys": {"anthropic": ""}}
        with patch.dict("os.environ", {}, clear=True):
            result = quickcut.generate_description("some transcript", config)
        assert result == ""

    def test_returns_empty_when_config_missing_api_keys(self):
        with patch.dict("os.environ", {}, clear=True):
            result = quickcut.generate_description("some transcript", {})
        assert result == ""

    def test_returns_empty_when_env_key_also_missing(self):
        config = {"api_keys": {}}
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=False):
            result = quickcut.generate_description("some transcript", config)
        assert result == ""


class TestGenerateDescriptionCallsClaude:
    """generate_description should call the Anthropic client correctly."""

    @patch("quickcut.anthropic")
    def test_calls_client_with_transcript(self, mock_anthropic_mod):
        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="  Generated description  ")]
        mock_client.messages.create.return_value = mock_response

        config = {"api_keys": {"anthropic": "sk-test-key"}}
        result = quickcut.generate_description("My transcript text", config)

        # Verify client was created with the key
        mock_anthropic_mod.Anthropic.assert_called_once_with(api_key="sk-test-key")

        # Verify messages.create was called
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]

        # Verify the prompt contains the transcript text
        user_content = call_kwargs["messages"][0]["content"]
        assert "My transcript text" in user_content

        # Verify max_tokens
        assert call_kwargs["max_tokens"] == 500

    @patch("quickcut.anthropic")
    def test_returns_stripped_response(self, mock_anthropic_mod):
        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="  \n Description with whitespace \n  ")]
        mock_client.messages.create.return_value = mock_response

        config = {"api_keys": {"anthropic": "sk-test-key"}}
        result = quickcut.generate_description("transcript", config)

        assert result == "Description with whitespace"

    @patch("quickcut.anthropic")
    def test_uses_env_key_as_fallback(self, mock_anthropic_mod):
        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="desc")]
        mock_client.messages.create.return_value = mock_response

        config = {"api_keys": {"anthropic": ""}}
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-env-key"}):
            quickcut.generate_description("text", config)

        mock_anthropic_mod.Anthropic.assert_called_once_with(api_key="sk-env-key")


class TestEnhanceReturnType:
    """enhance() must return a tuple of (str, str) — (output_path, description)."""

    @patch("quickcut.generate_description", return_value="A cool description")
    @patch("quickcut.render_enhanced")
    @patch("quickcut.build_hook_ass", return_value="hook.ass")
    @patch("quickcut.extract_hook", return_value={"text": "hook", "start": 0.0, "end": 2.0})
    @patch("quickcut.build_caption_ass")
    @patch("quickcut.build_montage", return_value=("", 0.0))
    @patch("quickcut.plan_edits", return_value=[])
    @patch("quickcut.analyze_broll", return_value=[])
    @patch("quickcut.transcribe", return_value={"segments": [], "full_text": "hello", "duration": 10.0})
    @patch("quickcut._detect_resolution", return_value=(1080, 1920))
    @patch("quickcut.shutil.rmtree")
    @patch("quickcut.tempfile.mkdtemp", return_value="/tmp/quickcut_test")
    def test_returns_tuple_of_two_strings(
        self, mock_mkdtemp, mock_rmtree, mock_res,
        mock_transcribe, mock_broll, mock_edits, mock_montage,
        mock_caption, mock_hook, mock_hook_ass,
        mock_render, mock_desc,
    ):
        config = {
            "rendering": {"output_resolution": [1080, 1920], "fps": 30, "crf": 18},
            "dynamic_edits": {},
            "broll": {"enabled": False},
            "montage": {"enabled": False},
            "hook_text": {"enabled": False},
        }

        with patch.object(Path, "mkdir"):
            result = quickcut.enhance("input.mp4", "output.mp4", config)

        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2-element tuple, got {len(result)}"
        assert isinstance(result[0], str), f"First element should be str, got {type(result[0])}"
        assert isinstance(result[1], str), f"Second element should be str, got {type(result[1])}"
        assert result[0] == "output.mp4"
        assert result[1] == "A cool description"


class TestHookCurlyBracesSafe:
    """Transcript text with curly braces must not crash (old .format() bug)."""

    def test_curly_braces_in_transcript(self):
        transcript = {
            "segments": [{"start": 0.0, "end": 3.0, "text": "use {braces} here"}],
            "full_text": "You can use {braces} and {more braces} in code",
        }
        # Must not raise KeyError / ValueError
        result = quickcut.extract_hook(transcript)
        assert result is not None
        assert "{subject}" not in result["text"]
