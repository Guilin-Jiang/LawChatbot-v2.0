# 使用aliyun
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000
EXPOSE 8501 
EXPOSE 11434

# 启动 FastAPI 和 Streamlit 服务
CMD ["bash", "run.sh"]
