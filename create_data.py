import openai
import os
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()  # .env ファイルを読み込む
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_faq_data(n=100, file_path="faq_data.txt"):
    prompt = f"""
    カスタマーサポートのFAQデータを {n} 件作成してください。
    各項目は「Q: 質問」「A: 回答」の形式で出力してください。
    """

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": "あなたはカスタマーサポートのFAQを作成するアシスタントです。"},
                  {"role": "user", "content": prompt}],
        max_tokens=2000
    )

    faq_data = response["choices"][0]["message"]["content"].strip()

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(faq_data)

    print(f"✅ {n} 件のFAQデータを `{file_path}` に保存しました！")

if __name__ == "__main__":
    generate_faq_data()
