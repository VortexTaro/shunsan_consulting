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

# 指示
各テキスト断片について、以下の**厳格なルール**に従って関連性を判断し、指定の形式で回答してください。

**厳格なルール:**
1.  テキスト断片が、ユーザーの質問の**主題に直接的**に答えているか？
    - 例: 質問が「オーダーノートの書き方」なら、断片は「書き方」について具体的に説明している必要がある。
2.  テキスト断片が、質問の主題から派生した**比喩的・哲学的・抽象的な内容**ではないか？
    - 例: 質問が「オーダーノート」についてなのに、断片が「周波数」や「宇宙」の**深い概念**について説明しているだけの場合、それは**関連していない ([IRRELEVANT])**とみなす。
3.  **例外:** ユーザーの質問自体が、その深い概念（例：「周波数とは何か？」）について問うている場合にのみ、ルール2の限りではない。

# 回答形式
- **本当に関連している場合:** `理由: [ルールに基づいた具体的な理由を1文で記述]`
- **関連していない場合:** `理由: [IRRELEVANT]`

番号を付けてリスト形式で出力し、他の余計な言葉は含めないでください。"""
    
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

# --- 初期設定 ---
st.title("いつでもしゅんさん")

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

            # 【絶対的フィルター】特定の条件下で、特定のファイルを除外する
            if "オーダーノート" in prompt and "周波数" not in prompt:
                docs_with_scores = [
                    (doc, score) for doc, score in docs_with_scores
                    if "周波数の正体.txt" not in doc.metadata.get('source', '')
                ]

            reasons = generate_source_reasons(prompt, docs_with_scores)
            
            relevant_sources = [
                {"doc": doc, "score": score, "reason": reason}
                for (doc, score), reason in zip(docs_with_scores, reasons) if reason != "[IRRELEVANT]"
            ]

            context = "\n\n".join([item["doc"].page_content for item in relevant_sources])
            
            # 関連情報の有無でAIの役割（プロンプト）を動的に切り替える
            if context:
                # 関連情報がある場合：忠実なアシスタント
                final_prompt = f"""あなたは親切なアシスタントです。以下の関連情報のみを使って、ユーザーの質問に誠実に答えてください。
--- 関連情報 ---
{context}
--- ユーザーの質問 ---
{prompt}"""
            else:
                # 関連情報がない場合：着眼点シフトモード
                final_prompt = f"""## モード2: 関連情報がない場合 (着眼点シフトモード)
- **君の役割:**
  - 君の役割は、ユーザーの悩みを直接的に解決することではありません。
  - その悩みが、「オーダーノート」の哲学全体から見ると、どのような**「素晴らしい機会」**や**「成長のサイン」**に見えるか、その**新しい「着眼点」**を提示し、ユーザーの視点を180度転換させることが、君の唯一の役割です。

- **思考プロセス:**
  1.  ユーザーの悩み（例：お金がない、人間関係が悪い）の表面的な事象を受け取ります。
  2.  次に、その事象の裏にある**本質的なテーマ**（例：価値の受け取り方、自己肯定感、理想の世界観）を、君が持つナレッジベース全体から見抜きます。
  3.  そして、そのテーマに基づいて、ユーザーに**本質的な問い**を投げかけます。

- **具体的な会話開始の例:**
  - **ユーザーの悩み:** 「今、お金がピンチなんです！」
  - **君の応答（悪い例）:** 「大変ですね。節約する方法や、収入を増やす方法を考えてみましょう。」
  - **君の応答（良い例）:** 「そっか、今、お金という形で、君にパワフルなメッセージが届いているんだね。そのピンチは、君が『自分には価値がない』って無意識に握りしめている古い思い込みを、手放すための最高のチャンスかもしれないよ。もし、そのピンチが『君の本当の価値に気づけ！』っていう宇宙からのサインだとしたら、何から始めてみたい？」

---
# その他の指示
ユーザーの質問に答えてください: {prompt}
"""
            
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