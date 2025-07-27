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

# --- 知識ベースの読み込みとインデックス構築 ---
@st.cache_resource(show_spinner="知識を構造化しています...")
def load_or_create_faiss_index(_embeddings):
    if os.path.exists(FAISS_INDEX_PATH):
        # 既存のインデックスをサイレントに読み込む
        return FAISS.load_local(FAISS_INDEX_PATH, _embeddings, allow_dangerous_deserialization=True)
    
    # 新しい知識ベースをサイレントに構築する
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
    # st.success("知識ベースの構築が完了しました！") # UI上のメッセージを削除
    return db

# --- 初期設定 ---
st.title("いつでもしゅんさん")

try:
    # APIキーの設定
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    # Embeddingモデルの初期化
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
except KeyError:
    st.error("GEMINI_API_KEYが設定されていません。Streamlit CloudのSecretsに設定してください。")
    st.stop()

# 知識ベースとメインモデルのロード
db = load_or_create_faiss_index(embeddings)
main_model = genai.GenerativeModel('gemini-2.5-pro')

# --- チャット履歴の初期化 ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "僕はしゅんさんのクローンです。あなたの質問に答えます。",
        "id": str(uuid.uuid4())
    }]

# --- UI処理 ---
# チャット履歴の表示
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=(AVATAR_IMAGE_PATH if msg["role"] == "assistant" else None)):
        st.markdown(msg["content"])
        # 参照元の表示 (シンプル版)
        if "sources" in msg and msg["sources"]:
            with st.expander("参照元ファイル"):
                # reasonはなくなったので、ファイルパスのみ表示
                for doc in msg["sources"]:
                    st.markdown(f"- `{doc.metadata.get('source', 'N/A')}`")

# ユーザーの入力処理
if prompt := st.chat_input("質問や相談したいことを入力してね"):
    st.session_state.messages.append({"role": "user", "content": prompt, "id": str(uuid.uuid4())})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant", avatar=AVATAR_IMAGE_PATH):
        placeholder = st.empty()
        
        with st.spinner("宇宙と通信中...ちょっとまってね..."):
            # 1. 知識ベースから関連情報を検索
            docs = db.similarity_search(prompt, k=5) # 少し多めに5件取得

            # 2. 【フィルター処理】特定のファイルを除外する
            # 初心者向けの質問(オーダーノート, 書き方)の場合、上級者向けファイル(周波数の正体.txt)を除外
            is_beginner_question = "オーダーノート" in prompt or "書き方" in prompt
            is_advanced_topic_asked = "周波数" in prompt and ("正体" in prompt or "詳しく" in prompt or "仕組み" in prompt)

            if is_beginner_question and not is_advanced_topic_asked:
                relevant_docs = [doc for doc in docs if "周波数の正体.txt" not in doc.metadata.get('source', '')]
            else:
                relevant_docs = docs
            
            # フィルター後の上位4件を使用
            relevant_docs = relevant_docs[:4]

            # 3. 会話履歴とプロンプトを構築
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            # 会話履歴をAPIが期待する形式に変換（最新のプロンプトは除く）
            history_for_model = []
            for msg in st.session_state.messages[:-1]:
                role = "model" if msg["role"] == "assistant" else "user"
                history_for_model.append({'role': role, 'parts': [msg['content']]})

            # 状況に応じてプロンプトテンプレートを読み込む
            if context and relevant_docs:
                # 関連情報がある場合
                with open("prompt_with_context.md", "r", encoding="utf-8") as f:
                    template = f.read()
                final_prompt = template.format(context=context, prompt=prompt)
            else:
                # 関連情報がない場合
                with open("prompt_without_context.md", "r", encoding="utf-8") as f:
                    template = f.read()
                final_prompt = template.format(prompt=prompt)
            
            # 4. AI応答生成（会話履歴を渡す）
            try:
                messages_for_api = history_for_model + [{'role': 'user', 'parts': [final_prompt]}]
                stream = main_model.generate_content(messages_for_api, stream=True)
                full_response = "".join(chunk.text for chunk in stream)
                placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"AI応答の生成中にエラーが発生しました: {e}"
                placeholder.error(full_response)

    # 4. アシスタントの応答と参照元を履歴に保存
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": relevant_docs, # 参照ドキュメントのリストをそのまま保存
        "id": str(uuid.uuid4())
    })
    st.rerun() 