"""gemini_utils の単体テスト（Streamlit / Whisper / 実APIには依存しない）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gemini_utils import (
    DEFAULT_GEMINI_MODEL,
    GEMINI_MODELS,
    build_prompt,
    generate_minutes,
    is_audio_or_video_file,
    is_text_file,
    is_vtt_file,
    is_word_file,
)


def test_default_model_is_in_model_list():
    assert DEFAULT_GEMINI_MODEL in GEMINI_MODELS
    # 廃止済みモデルが一覧に混入していないこと
    assert "gemini-pro" not in GEMINI_MODELS
    assert not any(m.startswith("gemini-1.5") for m in GEMINI_MODELS)


def test_build_prompt_replaces_only_known_placeholders():
    tpl = "日付: {date}\n本文: {transcript}\n例: {} や {unknown} はそのまま"
    out = build_prompt(tpl, "こんにちは", "2026年10月23日")
    assert out == "日付: 2026年10月23日\n本文: こんにちは\n例: {} や {unknown} はそのまま"


def test_build_prompt_does_not_crash_on_braces():
    # 旧実装の str.format では KeyError/ValueError になっていたケース
    tpl = "{transcript} の要約をJSON形式 {} で"
    assert build_prompt(tpl, "abc", "2026年") == "abc の要約をJSON形式 {} で"


def test_file_type_checks():
    assert is_audio_or_video_file("a.MP3") is True
    assert is_audio_or_video_file("a.mp4") is True
    assert is_audio_or_video_file("a.txt") is False
    assert is_text_file("a.vtt") is True
    assert is_text_file("a.md") is True
    assert is_word_file("a.docx") is True
    assert is_word_file("a.txt") is False
    assert is_vtt_file("a.VTT") is True
    assert is_vtt_file("a.txt") is False


class _FakeResponse:
    text = "生成された議事録"


class _FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, model, contents):
        self.calls.append((model, contents))
        return _FakeResponse()


class _FakeClient:
    def __init__(self):
        self.models = _FakeModels()


def test_generate_minutes_uses_default_model_and_built_prompt():
    client = _FakeClient()
    seen_keys = []

    def factory(api_key):
        seen_keys.append(api_key)
        return client

    out = generate_minutes(
        "文字起こし本文",
        "{date} の議事録: {transcript}",
        api_key="dummy",
        date_str="2026年10月23日",
        client_factory=factory,
    )
    assert out == "生成された議事録"
    assert seen_keys == ["dummy"]
    model, contents = client.models.calls[0]
    assert model == DEFAULT_GEMINI_MODEL
    assert contents == "2026年10月23日 の議事録: 文字起こし本文"


def test_generate_minutes_respects_explicit_model():
    client = _FakeClient()
    generate_minutes(
        "x", "{transcript}", api_key="k", model_name="gemini-2.5-pro",
        client_factory=lambda api_key: client,
    )
    assert client.models.calls[0][0] == "gemini-2.5-pro"
