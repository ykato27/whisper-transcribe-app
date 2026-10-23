"""
gemini_utils.py
Gemini連携とファイル種別判定など、UI（Streamlit）やWhisperに依存しない
純粋なロジックを集めたモジュール。単体テストはこちらを対象にする。

- Geminiのモデル一覧と既定値はここで一元管理する（二重管理の解消）。
- 旧SDK google-generativeai は廃止されたため、新SDK google-genai を使う。
- プロンプト組み立ては str.format を使わない。カスタムテンプレートに
  {transcript} / {date} 以外の波括弧（例文の {} など）が含まれていても
  KeyError / ValueError で落ちないようにするため。
"""

import os

# Geminiモデルの唯一の定義場所。アプリ側はこの定数だけを参照する。
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
]

GEMINI_MODEL_DESCRIPTIONS = {
    "gemini-2.5-flash": "高速・低コストの標準モデル（推奨）",
    "gemini-2.5-pro": "最高性能モデル（複雑な議事録向け）",
    "gemini-2.5-flash-lite": "最速・最安モデル（短い音声向け）",
}

AUDIO_VIDEO_EXTENSIONS = [
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".avi", ".mov", ".mkv",
]
TEXT_EXTENSIONS = [".txt", ".md", ".text", ".vtt"]
WORD_EXTENSIONS = [".docx", ".doc"]


def _ext(filename):
    return os.path.splitext(filename)[1].lower()


def is_audio_or_video_file(filename):
    """音声・動画ファイルかどうかを判定"""
    return _ext(filename) in AUDIO_VIDEO_EXTENSIONS


def is_text_file(filename):
    """テキストファイルかどうかを判定（VTTを含む）"""
    return _ext(filename) in TEXT_EXTENSIONS


def is_word_file(filename):
    """Wordファイルかどうかを判定"""
    return _ext(filename) in WORD_EXTENSIONS


def is_vtt_file(filename):
    """VTTファイルかどうかを判定"""
    return _ext(filename) == ".vtt"


def build_prompt(prompt_template, transcript, date_str):
    """プロンプトテンプレートへ {transcript} と {date} だけを差し込む。

    str.format と違い、それ以外の波括弧はそのまま残るため、
    ユーザーのカスタムテンプレートに {} 等が含まれても落ちない。
    """
    return prompt_template.replace("{transcript}", transcript).replace(
        "{date}", date_str
    )


def _default_client_factory(api_key):
    """新SDK（google-genai）のクライアントを作る。"""
    from google import genai

    return genai.Client(api_key=api_key)


def generate_minutes(transcript, prompt_template, api_key, model_name=None,
                     date_str="", client_factory=None):
    """Gemini APIで議事録を生成する。

    client_factory はテスト用の差し込み口（api_key を受け取り、
    models.generate_content(model=..., contents=...) を持つ
    クライアントを返す callable）。
    """
    if model_name is None:
        model_name = DEFAULT_GEMINI_MODEL
    factory = client_factory or _default_client_factory
    client = factory(api_key)
    prompt = build_prompt(prompt_template, transcript, date_str)
    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text
