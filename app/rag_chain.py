import faiss
import numpy as np
import os
import pickle
import requests
from transformers import AutoTokenizer, AutoModel
from . import load_documents

# 加载 HuggingFace 嵌入模型
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def encode_text(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()



# 从磁盘加载 FAISS 向量数据库 + metadata
def load_faiss_index(index_path="data/vector_index"):
    # 加载向量索引
    index = faiss.read_index(os.path.join(index_path, "index.faiss"))

    # 加载 metadata 信息（包含原文内容 + source）
    with open(os.path.join(index_path, "docs.pkl"), "rb") as f:
        documents = pickle.load(f)  # List[Document]

    return index, documents



# 使用 Ollama 运行 Llama 模型
def get_answer(query):
    host = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    model_mistral = os.getenv("OLLAMA_MODEL", "mistral")

    url = f"{host}/api/chat"

    payload = {
        "model": model_mistral,
        "messages": [
            {"role": "user", "content": query}
        ],
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("Ollama response:")
        print(response.text)

        data = response.json()  # ⚠️ 出错点
        return data["message"]["content"]
    
    except requests.exceptions.RequestException as e:
        print("Ollama 请求失败:", e)
        return "抱歉，无法连接 Ollama下的mistral 模型。"



def retrieve_and_generate(query, chat_history=None, top_k=3):
    # 加载向量库和原始文档
    index, documents = load_faiss_index()

    # 查询向量化
    query_embedding = encode_text([query])
    _, I = index.search(query_embedding, k=top_k)

    # 取出相似段落（可多个）
    similar_texts = []
    for i in I[0]:
        doc = documents[i]
        chunk_text = doc.page_content.strip()
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "?")
        similar_texts.append(f"[{source} - 页码 {page}] {chunk_text}")

    context = "\n\n".join(similar_texts)

    # 构造历史聊天记录（可选）
    history_prompt = ""
    if chat_history:
        for msg in chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ["user", "assistant"]:
                prefix = "用户" if role == "user" else "助手"
                history_prompt += f"{prefix}：{content}\n"

    # 构造完整提示词
    prompt = "\n".join([
    "你是一个专业法律助理，请基于以下聊天记录和资料回答用户问题。",
    f"\n聊天记录:\n{history_prompt}",
    f"\n资料:\n{context}",
    f"\n问题: {query}"
    ])


    answer = get_answer(prompt)

    return {
        "answer": answer,
        "sources": similar_texts  # 返回给前端引用展示
    }
