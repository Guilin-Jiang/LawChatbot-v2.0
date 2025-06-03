import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import TokenTextSplitter

def load_all_documents(folder_path="data/pdf_docs"):
    all_docs = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            loader = PyPDFLoader(filepath)
            docs = loader.load()  # 一页一个 Document
            
            # 添加来源元数据
            for doc in docs:
                doc.metadata["source"] = filename  # 如：constitution.pdf
            all_docs.extend(docs)

    return all_docs



def split_documents_by_sentence(docs, max_tokens=500, chunk_overlap=50):
    """
    尽可能保留句子完整性进行 chunk 切分
    """
    text_splitter = TokenTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(docs)



def embed_documents():
    documents = load_all_documents()
    split_docs_by_sent = split_documents_by_sentence(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs_by_sent, embeddings)
    vectorstore.save_local("data/vector_index")
    with open("data/vector_index/docs.pkl", "wb") as f:
        pickle.dump(split_docs_by_sent, f)


if __name__ == "__main__":
    embed_documents()