# 🎤 Whisper文字起こしツール

OpenAIのWhisperモデルを使用した音声文字起こしアプリ

## 機能

- 複数の音声形式に対応（MP3, WAV, M4A, OGG, FLAC, MP4）
- 複数の言語に対応（自動検出含む）
- リアルタイム進捗表示
- チャンク処理による大容量ファイル対応

## Streamlit Cloudへのデプロイ

1. GitHubにリポジトリを作成
2. 以下のファイルをコミット:
   - `app.py`
   - `requirements.txt`
   - `packages.txt`
   - `.streamlit/config.toml`
3. [Streamlit Cloud](https://streamlit.io/cloud)でデプロイ

## ローカル実行
依存関係は uv で管理しています（`pyproject.toml` / `uv.lock`）。
`requirements.txt` は Streamlit Cloud 用に uv から書き出した固定版です。

```bash
# 依存関係インストール（uv）
uv sync

# アプリ起動
uv run streamlit run app.py
```

pip を使う場合:
```bash
pip install -r requirements.txt

# FFmpegインストール（Ubuntu/Debian）
sudo apt-get install ffmpeg

# FFmpegインストール（macOS）
brew install ffmpeg

# アプリ起動
streamlit run app.py