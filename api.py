from fastapi import FastAPI
import openai
import os
import uvicorn
from dotenv import load_dotenv
from functions import load_faq_data, chunk_text, load_or_create_faiss_index, generate_answer_with_rag

# 環境設定
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# FAQデータをロード & チャンク分割
text_data = load_faq_data()
chunks = chunk_text(text_data)

# Faiss インデックスのロード（必要なら作成）
index = load_or_create_faiss_index(chunks)

# FastAPI のインスタンス作成
app = FastAPI()

# エンドポイント（質問を受け付けて回答を返す）
@app.get("/ask")
def ask_question(query: str):
    """
    クエリを受け取り、RAGでFAQ検索 & ChatGPT で回答を生成するAPI
    """
    answer = generate_answer_with_rag(query, index, chunks)
    return {"query": query, "answer": answer}

# ローカル実行（デプロイせずローカルで試す場合）
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)