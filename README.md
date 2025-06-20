# LawChatbot-v2.0(Streamlit+FastAPI+Langchain+RAG+OllamaMistral)
这是一个简易的法律助手chatbot，基于提供的美国宪法pdf，做RAG，并且能多轮Q&A和展示引用的本地文档。
用Streamlit做前端简易的UI交互，用fastapi做前端和后端的中间层API，用langchain作为chatbot架构，下载ollama下的mistial模型到本地。

#####
以后完善功能的方向
```text
| 功能类别     | 功能名称                       | 意义                        | 技术关键词                                                       |
| -------- | -------------------------- | ------------------------- | ----------------------------------------------------------- |
| 🧩 交互体验  | ✅ 用户反馈按钮                   | 用户可标记回答“满意 / 不满意”         | Streamlit按钮 + 本地日志记录                                        |
| 💬 控制生成  | ✅ 模型温度/长度控制                | UI允许用户设置回答风格（简洁/详细）       | Streamlit表单 + OpenAI/Mistral参数调节                            |
| 🛠️ 文本能力 | ✅ 答案重写/纠错                  | 对生成的答案重新措辞或修复语义逻辑         | 二次调用LLM + prompt改写                                          |
| 🔐 安全边界  | ✅ 限制非法问题                   | 防止用户问与法律无关的内容             | 规则判断 or 分类模型过滤                                              |
| 🧪 工程能力  | ✅ 自动化测试接口                  | 对 API 提交合法问题，验证输出         | FastAPI TestClient / pytest                                 |
| 🚀 性能优化  | ✅ 嵌入缓存                     | 对相同问题不重复嵌入，提高速度           | FAISS / SQLite 缓存向量或LangChain Retriever Cache               |
| 📅Google日历写入功能
```

## 上手试试吧！快速开始！
#### 需要提前下载
1.Docker(https://www.docker.com/)
2.官网下载Ollama(https://ollama.com/) 
  在terminal运行
  ```bash
  ollama run mistral
  ```

#### 准备文件
下载文件夹 CMD：
```bash
cd ./desktop
github clone https://github.com/Guilin-Jiang/LawChatbot-v2.0
```

#### 开始运行
```bash
cd ./law_chatbot
docker-compose up --build
```

#### 本地测试效果
open URL http://0.0.0.0:8501

## 项目文件概述
```text
law_chatbot/
├── app/
│   ├── main.py            # FastAPI 接口服务
│   ├── rag_chain.py       # RAG 构建逻辑
│   ├── load_documents.py  # 加载并嵌入法律文件
├── data/
│   ├── pdf_docs           # 法律文件的pdf
│   │   ├── us_constitution.pdf
│   │   ├── us_immigration_law.pdf
│   │   ├── uw_madison_rules.pdf
│   ├── vector_index       # Embedding的向量库
│   │   ├── docs.pkl
│   │   ├── index.pkl
│   │   ├── index.faiss
├── streamlit_app.py       # 前端页面逻辑
├── run.sh                 # 启动脚本
├── Dockerfile             # 构建镜像
├── docker-compose.yml     # 容器管理
├── requirements.txt       # 依赖列表
└── README.md              # 项目说明文档
```
