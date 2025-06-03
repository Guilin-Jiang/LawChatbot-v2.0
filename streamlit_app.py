import streamlit as st
import requests

st.set_page_config(page_title="多轮对话法律智能助手", layout="centered")
st.title("法律智能多轮问答助手")

# 初始化对话历史
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List[Dict]，用于传递给后端

# 展示历史对话
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
user_input = st.chat_input("请输入您的问题...")

if user_input:
    # 添加用户消息到聊天记录
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # 构造请求体
    payload = {
        "query": user_input,
        "chat_history": st.session_state.chat_history
    }

    try:
        response = requests.post("http://localhost:8000/ask", json=payload)

        if response.status_code == 200:
            data = response.json()
            answer = data["answer"]
            sources = data["sources"]

            # 添加 AI 回复到聊天记录
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            with st.chat_message("assistant"):
                st.markdown(answer)

                with st.expander("📚 引用段落"):
                    for src in sources:
                        st.markdown(f"- {src}")

        else:
            st.error(f"服务器错误：{response.status_code}")
    except Exception as e:
        st.error("连接失败，请检查 FastAPI 是否已启动")
        st.exception(e)
