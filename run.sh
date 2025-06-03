#!/bin/bash

# -------- 向量数据库构建 --------
if [ -f "data/vector_index/index.faiss" ] && [ -f "data/vector_index/docs.pkl" ]; then
  echo "FAISS 索引已存在，跳过构建"
else
  echo "开始构建向量数据库..."
  python app/load_documents.py
fi

# 启动 FastAPI 服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# 启动 Streamlit 前端
streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501