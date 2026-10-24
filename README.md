# 🎤 Whisper文字起こしツール

OpenAIのWhisperモデルを使った音声文字起こしアプリ（Streamlit）です。文字起こし結果は Google Gemini で要約・整形もできます。

## 機能

- 複数の音声形式に対応（MP3, WAV, M4A, OGG, FLAC, MP4）
- 複数の言語に対応（自動検出含む）
- リアルタイム進捗表示
- チャンク処理による大容量ファイル対応
- Geminiによる要約・整形（`gemini_utils.py`）
  - Google Gemini API（新SDK `google-genai`）を使用。要約などに使うプロンプトはカスタマイズ可能
  - モデルの既定値は `gemini_utils.py` の `DEFAULT_GEMINI_MODEL` で一元管理

## 構成

- `app.py` — Streamlitアプリ本体
- `gemini_utils.py` — UI非依存のGemini連携ロジック（単体テスト対象）
- `tests/` — `gemini_utils.py` の単体テストなど
- `pyproject.toml` / `uv.lock` — 依存関係（uv管理）
- `requirements.txt` — Streamlit Cloud用にuvから書き出した固定版
- `packages.txt` / `.streamlit/config.toml` — Streamlit Cloud用設定

## ローカル実行

依存関係は uv で管理しています。

```bash
# 依存関係インストール（uv）
uv sync

# アプリ起動
uv run streamlit run app.py
```

Gemini機能を使う場合は、Gemini APIキーをアプリの画面／環境設定から設定してください。

pipを使う場合:

```bash
pip install -r requirements.txt
streamlit run app.py
```

FFmpegが必要です（Ubuntu/Debian: `sudo apt-get install ffmpeg`、macOS: `brew install ffmpeg`）。

## テスト

```bash
uv run pytest
```

## Streamlit Cloudへのデプロイ

1. このリポジトリをGitHubにプッシュ
2. [Streamlit Cloud](https://streamlit.io/cloud) で `app.py` を指定してデプロイ（依存は `requirements.txt` と `packages.txt` を使用）
