import streamlit as st
import openai
import os
from dotenv import load_dotenv
from functions import load_faq_data, chunk_text, load_or_create_faiss_index, generate_answer_with_rag, search_similar_chunks

# 環境設定
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# FAQデータ読み込み & チャンク化
text_data = load_faq_data()
chunks = chunk_text(text_data)

# Faiss インデックスのロード（必要なら作成）
index = load_or_create_faiss_index(chunks)

# インデックスが None になるケースを防ぐ
if index is None:
    st.error("⚠️ Faiss インデックスのロードに失敗しました！")
    st.stop()
else:
    print("✅ 既存の Faiss インデックスをロードしました！")

st.title("FAQ 検索 & AIアシスタント")
query = st.text_input("質問を入力してください:")

if st.button("検索"):
    if query:
        try:
            answer = generate_answer_with_rag(query, index, chunks)
            st.write("🔹 **回答:**")
            st.write(answer)

            st.write("\n🔍 **参考情報（検索されたFAQ）:**")
            similar_chunks = search_similar_chunks(query, k=3, index=index, chunks=chunks)

            if not similar_chunks:
                st.warning("⚠️ 関連するFAQが見つかりませんでした。")
            else:
                for text, score in similar_chunks:
                    st.write(f"- {text} (スコア: {score:.4f})")

        except Exception as e:
            st.error(f"⚠️ エラーが発生しました: {str(e)}")

    else:
        st.warning("質問を入力してください！")
