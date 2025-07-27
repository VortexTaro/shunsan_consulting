import streamlit as st
import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import uuid

# --- 定数とパス設定 ---
KNOWLEDGE_BASE_DIR = "オーダーノート現実創造プログラム"
FAISS_INDEX_PATH = "data/faiss_index"
AVATAR_IMAGE_PATH = "assets/avatar.png"

# --- 関所機能：AIによる関連性チェック (Gemini版) ---
def generate_source_reasons(prompt, docs_with_scores):
    if not docs_with_scores:
        return []

    model = genai.GenerativeModel('gemini-2.5-pro') # モデルを2.5-proに統一
    content_list = [f"<{i+1}>\n{doc.page_content}\n</{i+1}>" for i, (doc, _) in enumerate(docs_with_scores)]
    formatted_chunks = "\n\n".join(content_list)

    system_prompt = "あなたは、ユーザーの質問と複数のテキスト断片の関係性を分析する専門家です。各テキスト断片がユーザーの質問に本当に関連しているかを厳密に判断し、関連している場合はその核心的な理由を、関連していない場合はその旨を明確に示してください。"
    user_message = f"""以下のユーザーの質問と、それに関連する可能性のあるテキスト断片リストを読んでください。
# ユーザーの質問: {prompt}
# テキスト断片リスト: {formatted_chunks}
# 指示: 各テキスト断片について、1. **関連している場合:** `理由: [具体的な理由]` 2. **関連していない場合:** `理由: [IRRELEVANT]` の形式で、番号を付けてリストで出力してください。"""
    
    try:
        response = model.generate_content(f"{system_prompt}\n{user_message}")
        reasons_text = response.text
        reasons = [line.split('理由: ', 1)[1].strip() for line in reasons_text.strip().split('\n') if '理由: ' in line]
        return reasons if len(reasons) == len(docs_with_scores) else ["[IRRELEVANT]"] * len(docs_with_scores)
    except Exception:
        return ["[IRRELEVANT]"] * len(docs_with_scores)

# --- 知識ベースの読み込みとインデックス構築 ---
@st.cache_resource(show_spinner="知識を構造化しています...")
def load_or_create_faiss_index(_embeddings):
    if os.path.exists(FAISS_INDEX_PATH):
        st.info("既存の知識ベースを読み込んでいます...")
        return FAISS.load_local(FAISS_INDEX_PATH, _embeddings, allow_dangerous_deserialization=True)
    
    st.info(f"新しい知識ベースを構築しています... (初回のみ)")
    if not os.path.isdir(KNOWLEDGE_BASE_DIR):
        st.error(f"知識フォルダ '{KNOWLEDGE_BASE_DIR}' が見つかりません。")
        st.stop()
        
    all_splits = []
    for filepath in glob.glob(f'{KNOWLEDGE_BASE_DIR}/**/*.txt', recursive=True):
        try:
            loader = TextLoader(filepath, encoding='utf-8')
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)
        except Exception as e:
            st.warning(f"ファイル処理中にエラー: {filepath} ({e})")

    if not all_splits:
        st.error("知識ベースからドキュメントを読み込めませんでした。")
        st.stop()

    db = FAISS.from_documents(all_splits, _embeddings)
    db.save_local(FAISS_INDEX_PATH)
    st.success("知識ベースの構築が完了しました！")
    return db

# --- デザイン設定 ---
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap');

/* 全体のフォントと背景色 (ヘッダー・フッター・ボトムエリア含む) */
body, .stApp, [data-testid="stHeader"], [data-testid="stFooter"], [data-testid="stBottom"] {
    font-family: 'Noto Sans JP', sans-serif;
    background-color: #1E1E1E !important; /* ダークグレーの背景 */
    color: #EAEAEA; /* 明るいグレーのテキスト */
}

/* アプリのタイトル */
h1 {
    color: #FFFFFF;
    text-shadow: 1px 1px 5px rgba(0,0,0,0.5);
}

/* チャットメッセージのスタイル */
div[data-testid="stChatMessage"] {
    background-color: #2D2D2D; /* やや明るいグレー */
    border-radius: 12px;
    border: 1px solid #444444;
}

div[data-testid="stChatMessage"] p {
    color: #EAEAEA;
}

/* チャット入力欄のコンテナ (フッター部分) */
div[data-testid="stChatInput"] {
    background-color: #1E1E1E !important;
    border-top: 1px solid #444444;
}

/* チャット書き込み欄 */
textarea[data-testid="stChatInputTextArea"] {
    background-color: #2D2D2D;
    color: #EAEAEA;
    border: 1px solid #555555;
    border-radius: 5px;
}

/* プレースホルダーのスタイル */
textarea[data-testid="stChatInputTextArea"]::placeholder {
  color: #888888;
}

/* スピナーのテキスト */
.stSpinner > div > div {
    color: #FFFFFF;
}
</style>
"""

# --- 初期設定 ---
st.title("いつでもしゅんさん")
st.markdown(custom_css, unsafe_allow_html=True)


try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
except KeyError:
    st.error("GEMINI_API_KEYが設定されていません。")
    st.stop()

db = load_or_create_faiss_index(embeddings)
main_model = genai.GenerativeModel('gemini-2.5-pro')

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "僕はしゅんさんのクローンです。あなたの質問に答えます。",
        "id": str(uuid.uuid4())
    }]

# --- UI処理 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=(AVATAR_IMAGE_PATH if msg["role"] == "assistant" else None)):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("参照元ファイル"):
                for item in msg["sources"]:
                    st.markdown(f"**`{item['doc'].metadata.get('source', 'N/A')}`** (理由: {item['reason']})")

if prompt := st.chat_input("質問や相談したいことを入力してね"):
    st.session_state.messages.append({"role": "user", "content": prompt, "id": str(uuid.uuid4())})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant", avatar=AVATAR_IMAGE_PATH):
        placeholder = st.empty()
        
        with st.spinner("思考中..."):
            docs_with_scores = db.similarity_search_with_score(prompt, k=5)
            reasons = generate_source_reasons(prompt, docs_with_scores)
            
            relevant_sources = [
                {"doc": doc, "score": score, "reason": reason}
                for (doc, score), reason in zip(docs_with_scores, reasons) if reason != "[IRRELEVANT]"
            ]

            context = "\n\n".join([item["doc"].page_content for item in relevant_sources])
            final_prompt = f"あなたは親切なアシスタントです。以下の関連情報のみを使って、ユーザーの質問に誠実に答えてください。\n\n--- 関連情報 ---\n{context if context else 'なし'}\n\n--- ユーザーの質問 ---\n{prompt}"
            
            try:
                stream = main_model.generate_content(final_prompt, stream=True)
                full_response = "".join(chunk.text for chunk in stream)
                placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"エラーが発生しました: {e}"
                placeholder.error(full_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": relevant_sources,
        "id": str(uuid.uuid4())
    })
    st.rerun() 