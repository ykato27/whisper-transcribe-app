#!/usr/bin/env python3
"""
Whisper文字起こし + Gemini議事録生成アプリ（設定保存対応版）
"""

import os
import tempfile
import whisper
import torch
import streamlit as st
import subprocess
import time
from pydub import AudioSegment
import math
import google.generativeai as genai
from datetime import datetime
import json

# ページ設定
st.set_page_config(
    page_title="AI議事録作成ツール", 
    page_icon="📝", 
    layout="wide"
)

# セッション状態の初期化
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""
if 'minutes' not in st.session_state:
    st.session_state.minutes = ""
if 'file_type' not in st.session_state:
    st.session_state.file_type = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'custom_prompts' not in st.session_state:
    st.session_state.custom_prompts = {}

# キャッシュ設定
@st.cache_resource
def load_whisper_model(model_name):
    """Whisperモデルをロード（キャッシュ）"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(model_name, device=device)

def check_ffmpeg():
    """FFmpegの存在確認"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def get_available_models():
    """利用可能なモデルリスト"""
    return ["tiny", "base", "small", "medium"]

def process_audio_chunk(model, audio_segment, language=None):
    """音声チャンクを処理"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as chunk_file:
        try:
            audio_segment.export(chunk_file.name, format="wav")
            options = {"language": language} if language else {}
            result = model.transcribe(chunk_file.name, **options)
            return result["text"]
        finally:
            try:
                os.unlink(chunk_file.name)
            except:
                pass

def get_default_prompt_templates():
    """デフォルトの議事録生成用プロンプトテンプレート"""
    return {
        "標準（詳細）": """以下の会議の文字起こしテキストから、詳細な議事録を作成してください。

# 文字起こしテキスト
{transcript}

# 出力形式
以下の形式でMarkdown形式の議事録を作成してください：

## 会議概要
- 日時: {date}
- 議題: [テキストから推測]

## 参加者
[発言から推測される参加者をリストアップ]

## 討議内容
### [トピック1]
- [要点1]
- [要点2]

### [トピック2]
- [要点1]
- [要点2]

## 決定事項
- [決定事項1]
- [決定事項2]

## アクションアイテム
- [ ] [担当者] [タスク内容] [期限]

## 次回の予定
[次回ミーティングについて言及があれば記載]

## 補足・メモ
[その他重要な情報]
""",
        
        "簡潔版": """以下の会議の文字起こしから、簡潔な議事録を作成してください。

# 文字起こしテキスト
{transcript}

# 出力形式
## 📅 {date}の会議まとめ

### 💡 主な議題
[3〜5個の箇条書き]

### ✅ 決定事項
[重要な決定のみ]

### 📋 TODO
- [ ] [タスク]

### 📌 メモ
[補足情報]
""",
        
        "ビジネス正式版": """以下は会議の文字起こしです。正式なビジネス文書として議事録を作成してください。

# 文字起こしテキスト
{transcript}

# 出力形式

# 議事録

**日時**: {date}
**議題**: [議題名]
**出席者**: [氏名をリストアップ]
**欠席者**: [該当あれば]
**記録者**: [推測または未記載]

---

## 1. 開会
[開会の挨拶や趣旨説明]

## 2. 前回議事録の確認
[前回の振り返りがあれば]

## 3. 議事
### 3.1 [議題1]
**説明**: [内容]
**討議**: [主な意見]
**結論**: [決定事項]

### 3.2 [議題2]
**説明**: [内容]
**討議**: [主な意見]
**結論**: [決定事項]

## 4. 決定事項
1. [決定事項1]
2. [決定事項2]

## 5. アクションアイテム
| 担当者 | タスク | 期限 |
|--------|--------|------|
| [名前] | [内容] | [日付] |

## 6. 次回予定
**日時**: [次回の日時]
**議題**: [次回の議題]

## 7. 閉会

---
以上
"""
    }

def get_all_prompt_templates():
    """デフォルト + カスタムプロンプトを統合"""
    templates = get_default_prompt_templates()
    templates.update(st.session_state.custom_prompts)
    return templates

def generate_minutes_with_gemini(transcript, prompt_template, api_key):
    """Gemini APIで議事録を生成"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        current_date = datetime.now().strftime("%Y年%m月%d日")
        prompt = prompt_template.format(transcript=transcript, date=current_date)
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        raise Exception(f"Gemini API エラー: {str(e)}")

def is_audio_or_video_file(filename):
    """音声・動画ファイルかどうかを判定"""
    audio_video_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.mp4', '.avi', '.mov', '.mkv']
    ext = os.path.splitext(filename)[1].lower()
    return ext in audio_video_extensions

def is_text_file(filename):
    """テキストファイルかどうかを判定"""
    text_extensions = ['.txt', '.md', '.text']
    ext = os.path.splitext(filename)[1].lower()
    return ext in text_extensions

def transcribe_audio(uploaded_file, model_option, language_option):
    """音声ファイルの文字起こし処理"""
    
    file_size_mb = uploaded_file.size / (1024 * 1024)
    
    st.info(f"📄 **{uploaded_file.name}** ({file_size_mb:.2f} MB)")
    
    if file_size_mb > 200:
        st.error("ファイルサイズが200MBを超えています")
        return None
    
    # 音声プレビュー
    try:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext in ['mp3', 'wav', 'm4a', 'ogg']:
            st.audio(uploaded_file, format=f"audio/{file_ext}")
    except:
        st.warning("プレビューを表示できません")

    # 一時ファイルとして保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_filename = tmp_file.name
    
    try:
        # モデルロード
        with st.spinner("🔄 モデルをロード中..."):
            load_start = time.time()
            model = load_whisper_model(model_option)
            load_time = time.time() - load_start
            st.success(f"✅ モデルロード完了（{load_time:.2f}秒）")

        # 音声読み込み
        with st.spinner("📊 音声ファイルを解析中..."):
            audio = AudioSegment.from_file(temp_filename)
            duration_sec = audio.duration_seconds
            st.info(f"⏱️ 音声長: {duration_sec:.1f}秒")

        # チャンク分割設定（10秒単位）
        chunk_size_sec = 10
        num_chunks = math.ceil(duration_sec / chunk_size_sec)
        
        # 進捗バー
        progress_bar = st.progress(0)
        status_text = st.empty()
        transcribed_text = ""
        start_time = time.time()

        # チャンクごとに処理
        for i in range(num_chunks):
            start_ms = i * chunk_size_sec * 1000
            end_ms = min((i + 1) * chunk_size_sec * 1000, len(audio))
            chunk_audio = audio[start_ms:end_ms]

            # 文字起こし実行
            chunk_text = process_audio_chunk(
                model, 
                chunk_audio, 
                language=language_option if language_option else None
            )
            transcribed_text += chunk_text + " "

            # 進捗更新
            progress = (i + 1) / num_chunks
            elapsed = time.time() - start_time
            eta = (elapsed / progress - elapsed) if progress > 0 else 0
            
            progress_bar.progress(progress)
            status_text.text(
                f"📝 処理中: {progress*100:.1f}% | "
                f"残り予想時間: {eta:.1f}秒"
            )

        # 完了
        total_time = time.time() - start_time
        progress_bar.progress(1.0)
        status_text.empty()
        
        st.success(f"🎉 文字起こし完了！（処理時間: {total_time:.2f}秒）")
        
        return transcribed_text.strip()

    except Exception as e:
        st.error(f"❌ エラーが発生しました: {str(e)}")
        st.exception(e)
        return None
    
    finally:
        try:
            os.unlink(temp_filename)
        except:
            pass

def show_settings_page():
    """設定ページを表示"""
    st.title("⚙️ 設定")
    
    # API Key設定
    st.markdown("## 🔑 Gemini API Key")
    st.markdown("""
    Gemini APIを使用するには、API Keyが必要です。  
    [Google AI Studio](https://aistudio.google.com/app/apikey) から無料で取得できます（無料枠あり）。
    """)
    
    api_key_input = st.text_input(
        "API Keyを入力",
        value=st.session_state.api_key,
        type="password",
        help="入力したAPI Keyはセッション中のみ保持されます"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("💾 保存", type="primary"):
            st.session_state.api_key = api_key_input
            st.success("✅ API Keyを保存しました")
    with col2:
        if st.button("🗑️ クリア"):
            st.session_state.api_key = ""
            st.info("API Keyをクリアしました")
            st.rerun()
    
    if st.session_state.api_key:
        st.success("✅ API Key設定済み")
    else:
        st.warning("⚠️ API Keyが未設定です")
    
    # カスタムプロンプト設定
    st.markdown("---")
    st.markdown("## 📝 カスタムプロンプト管理")
    
    st.markdown("""
    独自の議事録フォーマットを作成できます。  
    `{transcript}` と `{date}` を使用できます。
    """)
    
    # 新規プロンプト追加
    with st.expander("➕ 新しいプロンプトを追加"):
        new_prompt_name = st.text_input("プロンプト名", key="new_prompt_name")
        new_prompt_content = st.text_area(
            "プロンプト内容",
            height=300,
            placeholder="""例:
以下の文字起こしから、技術ミーティングの議事録を作成してください。

# 文字起こし
{transcript}

# 要件
- 技術的な決定事項を明確に
- アクションアイテムを整理
- 次回の議題を抽出

日時: {date}
""",
            key="new_prompt_content"
        )
        
        if st.button("追加", type="primary"):
            if new_prompt_name and new_prompt_content:
                if new_prompt_name in get_default_prompt_templates():
                    st.error("❌ デフォルトのプロンプト名は使用できません")
                elif new_prompt_name in st.session_state.custom_prompts:
                    st.error("❌ 同じ名前のプロンプトが既に存在します")
                else:
                    st.session_state.custom_prompts[new_prompt_name] = new_prompt_content
                    st.success(f"✅ プロンプト「{new_prompt_name}」を追加しました")
                    st.rerun()
            else:
                st.error("プロンプト名と内容を入力してください")
    
    # 既存のカスタムプロンプト表示
    if st.session_state.custom_prompts:
        st.markdown("### 📋 保存済みカスタムプロンプト")
        
        for name, content in st.session_state.custom_prompts.items():
            with st.expander(f"📄 {name}"):
                st.code(content, language="markdown")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    # 編集機能
                    edit_key = f"edit_{name}"
                    if st.button(f"✏️ 編集", key=f"btn_edit_{name}"):
                        st.session_state[f"editing_{name}"] = True
                
                with col2:
                    # 削除機能
                    if st.button(f"🗑️ 削除", key=f"btn_delete_{name}"):
                        del st.session_state.custom_prompts[name]
                        st.success(f"削除しました: {name}")
                        st.rerun()
                
                # 編集モード
                if st.session_state.get(f"editing_{name}", False):
                    edited_content = st.text_area(
                        "編集",
                        value=content,
                        height=300,
                        key=f"edit_area_{name}"
                    )
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("💾 保存", key=f"save_{name}"):
                            st.session_state.custom_prompts[name] = edited_content
                            st.session_state[f"editing_{name}"] = False
                            st.success("保存しました")
                            st.rerun()
                    with col2:
                        if st.button("❌ キャンセル", key=f"cancel_{name}"):
                            st.session_state[f"editing_{name}"] = False
                            st.rerun()
    else:
        st.info("カスタムプロンプトはまだ追加されていません")
    
    # プロンプトのインポート/エクスポート
    st.markdown("---")
    st.markdown("## 📦 インポート/エクスポート")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📥 エクスポート")
        if st.session_state.custom_prompts:
            export_data = json.dumps(st.session_state.custom_prompts, ensure_ascii=False, indent=2)
            st.download_button(
                label="カスタムプロンプトをダウンロード",
                data=export_data,
                file_name="custom_prompts.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.info("エクスポートするプロンプトがありません")
    
    with col2:
        st.markdown("### 📤 インポート")
        uploaded_json = st.file_uploader(
            "JSONファイルをアップロード",
            type=["json"],
            help="以前エクスポートしたプロンプトをインポート"
        )
        
        if uploaded_json:
            try:
                imported_data = json.loads(uploaded_json.read())
                if st.button("インポート実行", type="primary"):
                    st.session_state.custom_prompts.update(imported_data)
                    st.success(f"✅ {len(imported_data)}個のプロンプトをインポートしました")
                    st.rerun()
            except json.JSONDecodeError:
                st.error("❌ 無効なJSONファイルです")

def show_main_page():
    """メインページを表示"""
    st.title("📝 AI議事録作成ツール")
    st.markdown("""
    **🎤 音声/動画ファイル** → Whisperで文字起こし → Geminiで議事録生成  
    **📄 テキストファイル** → 直接Geminiで議事録生成
    
    ---
    """)

    # API Key確認
    if not st.session_state.api_key:
        st.warning("⚠️ Gemini API Keyが設定されていません")
        st.info("👈 サイドバーの「設定」からAPI Keyを設定してください")
    
    # サイドバー設定
    st.sidebar.title("⚙️ Whisper設定")
    
    # ファイルアップロード
    st.markdown("### 📂 ファイルをアップロード")
    
    uploaded_file = st.file_uploader(
        "音声/動画ファイル または テキストファイルを選択",
        type=["mp3", "wav", "m4a", "ogg", "flac", "mp4", "avi", "mov", "mkv", "txt", "md", "text"],
        help="音声/動画: 文字起こし → 議事録生成 | テキスト: 直接議事録生成"
    )
    
    if uploaded_file is None:
        st.info("👆 ファイルをアップロードしてください")
        
        # 説明
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            #### 🎤 音声/動画ファイル
            - MP3, WAV, M4A, OGG, FLAC
            - MP4, AVI, MOV, MKV
            - 自動で文字起こし → 議事録生成
            """)
        with col2:
            st.markdown("""
            #### 📄 テキストファイル
            - TXT, MD (Markdown)
            - 既に文字起こし済みのテキスト
            - 直接議事録生成
            """)
        return

    # ファイルタイプを判定
    filename = uploaded_file.name
    
    if is_audio_or_video_file(filename):
        # ==================================
        # 音声/動画ファイルの処理
        # ==================================
        st.session_state.file_type = "audio_video"
        
        st.success("🎤 音声/動画ファイルを検出しました")
        st.info("📝 Whisperで文字起こしを実行します")
        
        # Whisper設定
        model_option = st.sidebar.selectbox(
            "モデルサイズ",
            get_available_models(),
            index=1,
            help="小さいモデルほど高速ですが精度は低下します"
        )
        
        model_info = {
            "tiny": "最速・最小（39M）",
            "base": "高速・小型（74M）",
            "small": "バランス型（244M）",
            "medium": "高精度（769M）",
        }
        st.sidebar.caption(f"💡 {model_info.get(model_option, '')}")
        
        language_option = st.sidebar.selectbox(
            "言語",
            options=["", "en", "ja", "zh", "de", "fr", "es", "ko", "ru"],
            index=2,  # デフォルト日本語
            format_func=lambda x: {
                "": "🌐 自動検出", 
                "en": "🇺🇸 英語", 
                "ja": "🇯🇵 日本語", 
                "zh": "🇨🇳 中国語",
                "de": "🇩🇪 ドイツ語", 
                "fr": "🇫🇷 フランス語", 
                "es": "🇪🇸 スペイン語",
                "ko": "🇰🇷 韓国語", 
                "ru": "🇷🇺 ロシア語"
            }.get(x, x)
        )

        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.sidebar.info(f"🖥️ デバイス: **{device}**")
        
        # FFmpegチェック
        if not check_ffmpeg():
            st.error("⚠️ FFmpegが利用できません。")
            st.stop()
        
        st.markdown("---")
        
        # 文字起こし実行
        if st.button("🚀 文字起こし開始", type="primary", use_container_width=True):
            transcribed_text = transcribe_audio(uploaded_file, model_option, language_option)
            
            if transcribed_text:
                st.session_state.transcribed_text = transcribed_text
                
                st.markdown("---")
                st.markdown("### 📄 文字起こし結果")
                
                st.text_area(
                    "テキスト",
                    value=st.session_state.transcribed_text,
                    height=300,
                    key="transcript_display"
                )
                
                st.download_button(
                    label="💾 文字起こしテキストをダウンロード",
                    data=st.session_state.transcribed_text,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
                st.success("✅ 文字起こし完了！下にスクロールして議事録を生成してください")
    
    elif is_text_file(filename):
        # ==================================
        # テキストファイルの処理
        # ==================================
        st.session_state.file_type = "text"
        
        st.success("📄 テキストファイルを検出しました")
        st.info("🤖 直接Geminiで議事録を生成します")
        
        # テキスト読み込み
        try:
            text_content = uploaded_file.read().decode('utf-8')
            st.session_state.transcribed_text = text_content
            
            st.markdown("---")
            st.markdown("### 📄 読み込んだテキスト")
            
            st.text_area(
                "内容",
                value=text_content,
                height=300,
                key="loaded_text_display"
            )
            
            st.success(f"✅ テキストを読み込みました（{len(text_content)}文字）")
            
        except Exception as e:
            st.error(f"❌ テキストの読み込みに失敗しました: {str(e)}")
            return
    
    else:
        st.error("❌ サポートされていないファイル形式です")
        return
    
    # ==================================
    # 議事録生成セクション（共通）
    # ==================================
    if st.session_state.transcribed_text:
        st.markdown("---")
        st.markdown("## 🤖 議事録を生成")
        
        # API Key確認
        if not st.session_state.api_key:
            st.error("⚠️ Gemini API Keyが設定されていません")
            st.info("👈 サイドバーの「設定」からAPI Keyを設定してください")
            return
        
        # プロンプトテンプレート選択
        st.markdown("### 📋 議事録のスタイル選択")
        
        templates = get_all_prompt_templates()
        template_names = list(templates.keys())
        
        selected_template_name = st.selectbox(
            "テンプレート",
            template_names,
            help="用途に応じて選択してください"
        )
        
        # プレビュー表示
        with st.expander("📝 プロンプトプレビュー"):
            st.code(templates[selected_template_name], language="markdown")
        
        # 議事録生成ボタン
        st.markdown("---")
        if st.button("✨ 議事録を生成", type="primary", use_container_width=True):
            with st.spinner("🤖 Geminiが議事録を生成中..."):
                try:
                    start_time = time.time()
                    
                    minutes = generate_minutes_with_gemini(
                        st.session_state.transcribed_text,
                        templates[selected_template_name],
                        st.session_state.api_key
                    )
                    
                    generation_time = time.time() - start_time
                    st.session_state.minutes = minutes
                    
                    st.success(f"✅ 議事録生成完了！（{generation_time:.2f}秒）")
                    
                except Exception as e:
                    st.error(f"❌ エラー: {str(e)}")
                    st.info("💡 API Keyが正しいか、設定ページで確認してください")
                    return
        
        # 生成された議事録を表示
        if st.session_state.minutes:
            st.markdown("---")
            st.markdown("## 📄 生成された議事録")
            
            # タブで表示切り替え
            view_tab1, view_tab2 = st.tabs(["📖 プレビュー", "📝 編集"])
            
            with view_tab1:
                st.markdown(st.session_state.minutes)
            
            with view_tab2:
                edited_minutes = st.text_area(
                    "議事録を編集",
                    value=st.session_state.minutes,
                    height=500,
                    key="edit_minutes"
                )
                
                if st.button("💾 編集を保存", type="secondary"):
                    st.session_state.minutes = edited_minutes
                    st.success("✅ 保存しました")
            
            # ダウンロードボタン
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Markdownでダウンロード",
                    data=st.session_state.minutes,
                    file_name=f"議事録_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            with col2:
                st.download_button(
                    label="📥 テキストでダウンロード",
                    data=st.session_state.minutes,
                    file_name=f"議事録_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

def main():
    """メイン関数"""
    # サイドバーでページ選択
    page = st.sidebar.radio(
        "ナビゲーション",
        ["🏠 ホーム", "⚙️ 設定"],
        label_visibility="collapsed"
    )
    
    if page == "🏠 ホーム":
        show_main_page()
    else:
        show_settings_page()

if __name__ == "__main__":
    main()