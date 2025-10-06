#!/usr/bin/env python3
"""
Whisper文字起こしWebアプリ（Streamlit Cloud対応版）
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

# ページ設定
st.set_page_config(
    page_title="Whisper文字起こしツール", 
    page_icon="🎤", 
    layout="wide"
)

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
    # Streamlit Cloudではメモリ制限があるため、小〜中サイズを推奨
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
            # ファイル削除を試行（失敗しても続行）
            try:
                os.unlink(chunk_file.name)
            except:
                pass

def main():
    st.title("🎤 Whisper文字起こしツール")
    st.markdown("""
    OpenAIのWhisperモデルを使用して、音声ファイルをテキストに変換します。
    
    **サポート形式**: MP3, WAV, M4A, OGG, FLAC, MP4
    """)

    # FFmpegチェック
    if not check_ffmpeg():
        st.error("⚠️ FFmpegが利用できません。システム管理者に連絡してください。")
        st.info("ローカル実行の場合: `apt-get install ffmpeg` または `brew install ffmpeg`")
        st.stop()
    
    # サイドバー設定
    st.sidebar.title("⚙️ 設定")
    
    model_option = st.sidebar.selectbox(
        "モデルサイズ",
        get_available_models(),
        index=1,
        help="小さいモデルほど高速ですが精度は低下します"
    )
    
    # モデルサイズの説明
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
        index=0,
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

    # デバイス情報
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.sidebar.info(f"🖥️ デバイス: **{device}**")
    
    if device == "CPU":
        st.sidebar.warning("⚡ CPUモードで動作中。処理に時間がかかる場合があります。")
    
    # ファイルアップロード
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "音声ファイルをアップロード",
        type=["mp3", "wav", "m4a", "ogg", "flac", "mp4"],
        help="最大200MBまで"
    )
    
    if uploaded_file is None:
        st.info("👆 音声ファイルをアップロードしてください")
        return

    # ファイル情報表示
    file_size_mb = uploaded_file.size / (1024 * 1024)
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(f"📄 **{uploaded_file.name}** ({file_size_mb:.2f} MB)")
    
    with col2:
        if file_size_mb > 200:
            st.error("ファイルが大きすぎます")
            return
    
    # 音声プレビュー
    try:
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
    except:
        st.warning("音声プレビューを表示できません")

    # 文字起こし実行
    if st.button("🚀 文字起こし開始", type="primary", use_container_width=True):
        
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

            # 結果表示
            st.markdown("---")
            st.markdown("### 📄 文字起こし結果")
            
            st.text_area(
                "テキスト",
                value=transcribed_text.strip(),
                height=300,
                label_visibility="collapsed"
            )
            
            # ダウンロードボタン
            st.download_button(
                label="💾 テキストをダウンロード",
                data=transcribed_text.strip(),
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt",
                mime="text/plain",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"❌ エラーが発生しました: {str(e)}")
            st.exception(e)
        
        finally:
            # 一時ファイル削除
            try:
                os.unlink(temp_filename)
            except:
                pass

if __name__ == "__main__":
    main()