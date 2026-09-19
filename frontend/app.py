"""Streamlit 演示前端（W9 完善）。

运行：streamlit run frontend/app.py
三栏规划：左对话 / 右推理 trace（层级、模型、工具序列、延迟、tokens）/ 底部预测账本看板。
"""

import httpx
import streamlit as st

API_BASE = "http://localhost:9000"

st.set_page_config(page_title="Sports Agent", layout="wide")
st.title("⚽ 推理感知型体育预测 Agent")

col_chat, col_trace = st.columns([3, 2])

with col_chat:
    query = st.chat_input("试试：英超积分榜 / 曼联对利物浦谁会赢 / 详细分析双红会")
    if query:
        with st.spinner("Agent 思考中…"):
            try:
                resp = httpx.post(
                    f"{API_BASE}/analyze", json={"query": query}, timeout=180
                ).json()
            except httpx.ConnectError:
                st.error("无法连接 FastAPI（localhost:9000），请先启动 agent runtime。")
                st.stop()

        st.markdown(f"**路由：{resp.get('level', '?')} → {resp.get('route', '?')}**")
        st.write(resp.get("answer", "(无回答)"))

with col_trace:
    st.subheader("推理 trace")
    st.caption("W9：此处展示节点、工具调用、延迟与 token 用量（骨架阶段仅 LLM 节点）")
    if query and "trace" in locals() and isinstance(resp, dict):
        st.json(resp.get("trace", []))
