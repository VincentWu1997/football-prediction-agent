"""Streamlit 演示前端（W9）。

三栏布局：
- 左栏：对话（连 FastAPI /analyze）
- 右栏：推理 trace（L 级别、模型/量化档、工具调用序列、每步耗时）
- 底栏：预测账本看板（pred_ledger 汇总：总数、准确率、LogLoss/Brier/RPS）

运行：streamlit run frontend/app.py
"""

import httpx
import streamlit as st

from sports_agent.eval.settlement import ledger_summary, settle_unsettled

API_BASE = "http://localhost:9000"

st.set_page_config(page_title="Sports Agent", layout="wide")
st.title("⚽ 推理感知型体育预测 Agent")

# ---- 初始化会话状态 ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_trace" not in st.session_state:
    st.session_state.last_trace = []

# ---- 顶部：对话区 ----
col_chat, col_trace = st.columns([3, 2])

with col_chat:
    st.subheader("对话")
    # 历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("试试：英超积分榜 / 曼联对利物浦谁会赢 / 详细分析双红会")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Agent 思考中…"):
                try:
                    resp = httpx.post(
                        f"{API_BASE}/analyze", json={"query": query}, timeout=180
                    ).json()
                except httpx.ConnectError:
                    st.error("无法连接 FastAPI（localhost:9000），请先启动 agent runtime。")
                    st.stop()

            level = resp.get("level", "?")
            route = resp.get("route", "?")
            answer = resp.get("answer", "(无回答)")
            trace = resp.get("trace", [])

            st.markdown(f"**路由：{level} → {route}**")
            st.write(answer)
            st.session_state.last_trace = trace

        st.session_state.messages.append({
            "role": "assistant",
            "content": f"**路由：{level} → {route}**\n\n{answer}",
        })

# ---- 右栏：推理 trace ----
with col_trace:
    st.subheader("推理 Trace")
    trace = st.session_state.last_trace
    if not trace:
        st.caption("发送一条消息后，此处展示 Planner/LLM/工具调用序列与耗时。")
    else:
        for i, step in enumerate(trace):
            node = step.get("node", "?")
            elapsed = step.get("elapsed_s", "?")
            with st.expander(f"步骤 {i+1}：{node}  ({elapsed}s)", expanded=i < 3):
                # 工具调用信息
                if step.get("tool_calls"):
                    for tc in step["tool_calls"]:
                        st.write(f"🔧 **{tc.get('name', '?')}**")
                        if tc.get("args"):
                            st.json(tc["args"], expanded=False)
                        if tc.get("result_summary"):
                            st.caption(tc["result_summary"])
                # LLM 输出摘要
                if step.get("llm_output"):
                    st.text(step["llm_output"][:200] + "…")
                # token 用量
                if step.get("tokens"):
                    st.caption(f"Tokens: {step['tokens']}")
                # 推理后端
                if step.get("model"):
                    st.caption(f"模型: {step['model']}")

# ---- 底栏：预测账本看板 ----
st.divider()
st.subheader("预测账本看板")

try:
    summary = ledger_summary()
except Exception as e:
    st.warning(f"无法连接 PostgreSQL（pred_ledger 不可用）：{e}")
    summary = {"total": 0, "unsettled": 0, "settled": 0,
               "avg_log_loss": None, "avg_brier": None, "avg_rps": None,
               "accuracy": None}

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
col_m1.metric("总预测数", summary["total"])
col_m2.metric("已结算", summary["settled"])
col_m3.metric("未结算", summary["unsettled"])
col_m4.metric("准确率",
              f"{summary['accuracy']:.1%}" if summary["accuracy"] is not None else "—")
col_m5.metric("Avg LogLoss",
              f"{summary['avg_log_loss']:.4f}" if summary["avg_log_loss"] is not None else "—")

col_m6, col_m7, col_m8 = st.columns(3)
col_m6.metric("Avg Brier",
              f"{summary['avg_brier']:.4f}" if summary["avg_brier"] is not None else "—")
col_m7.metric("Avg RPS",
              f"{summary['avg_rps']:.4f}" if summary["avg_rps"] is not None else "—")
col_m8.metric("赔率 baseline",
              "未测", help="W10: 赔率去水 baseline 对照")

# 回填按钮
col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    if st.button("回填结算", help="从 f_matches 查实际结果，计算 LogLoss/Brier/RPS 回写"):
        with st.spinner("结算中…"):
            result = settle_unsettled()
        st.success(
            f"已结算 {result['settled']} 条，"
            f"跳过 {result['skipped_no_result']} 条（无结果），"
            f"错误 {result['errors']} 条"
        )
        st.rerun()
