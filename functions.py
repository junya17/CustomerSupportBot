import faiss
import os
import openai
from sentence_transformers import SentenceTransformer

INDEX_PATH = "faiss_index.bin"
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def save_faiss_index(index):
    faiss.write_index(index, INDEX_PATH)

def load_faiss_index():
    """
    既存の Faiss インデックスをロードする。
    インデックスが存在しない場合は None を返す。
    """
    if os.path.exists(INDEX_PATH):
        print("✅ 既存の Faiss インデックスをロードしました。")
        return faiss.read_index(INDEX_PATH)
    else:
        print("⚠️ Faiss インデックスが見つかりません！")
        return None


def load_faq_data(file_path="faq_data.txt"):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
    
def chunk_text(text):
    """
    質問と回答を1セットとして分割するチャンク化関数。
    """
    chunks = []
    qa_pairs = text.strip().split("\n\n")  # FAQごとに分割（質問と回答の間は空行）
    
    for qa in qa_pairs:
        if "Q:" in qa and "A:" in qa:  # 質問と回答が両方含まれている場合のみ追加
            chunks.append(qa.strip())
    
    return chunks

def search_similar_chunks(query, k=5, index=None, chunks=[]):
    """
    クエリに対して類似したFAQチャンクを検索
    """
    query_vector = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vector, k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1 or idx >= len(chunks):  # インデックスが無効な場合を防ぐ
            continue
        chunk_text = chunks[idx]
        results.append((chunk_text, score))

    if not results:
        return [("該当するFAQが見つかりませんでした。", 0.0)]  # ✅ 結果がない場合のデフォルト値

    return results


def needs_reindexing(file_path="faq_data.txt", index_path=INDEX_PATH):
    """
    FAQデータが更新されたかチェックし、必要ならFaissを再構築
    """
    if not os.path.exists(index_path):
        return True  # インデックスがなければ再構築

    faq_mtime = os.path.getmtime(file_path)
    index_mtime = os.path.getmtime(index_path)

    return faq_mtime > index_mtime  # FAQデータの方が新しければ True を返す

def load_or_create_faiss_index(chunks):
    """
    既存の Faiss インデックスをロードし、必要なら再構築
    """
    index = load_faiss_index()
    
    if index is None:  # インデックスがない場合は新規作成
        print("🔄 Faiss インデックスを新規作成します...")
        dim = model.encode([chunks[0]], convert_to_numpy=True).shape[1]
        index = faiss.IndexFlatIP(dim)
        chunk_vectors = model.encode(chunks, convert_to_numpy=True)
        index.add(chunk_vectors)
        save_faiss_index(index)
        print("✅ Faiss インデックスの作成 & 保存が完了しました！")
    else:
        print("✅ 既存の Faiss インデックスを使用します。")

    return index



def generate_answer_with_rag(query, index, chunks):
    retrieved_chunks = search_similar_chunks(query, k=5, index=index, chunks=chunks)
    
    if not retrieved_chunks or retrieved_chunks[0][0] == "該当するFAQが見つかりませんでした。":
        return "該当するFAQが見つかりませんでした。"

    context = "\n".join([f"- {text}（スコア: {score:.4f}）" for text, score in retrieved_chunks])

    prompt = f"""
    以下の情報を参考にして、質問に答えてください。
    ただし、数値や日付はできるだけそのまま保持してください。

    【参考情報】
    {context}

    【質問】
    {query}

    【回答】
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "あなたは知識豊富なアシスタントです。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        return response["choices"][0]["message"]["content"]

    except Exception as e:
        return f"⚠️ ChatGPT の応答中にエラーが発生しました: {str(e)}"
