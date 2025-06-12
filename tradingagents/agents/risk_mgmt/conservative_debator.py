from langchain_core.messages import AIMessage
import time
import json


def create_safe_debator(llm):
    def safe_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        safe_history = risk_debate_state.get("safe_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""作為『保守型』風險分析師，你的首要目標是保護資產、降低波動並確保穩定可靠的成長。你重視穩定、安全與風險控管，謹慎評估潛在損失、經濟衰退及市場波動。當評估交易員的決策或計畫時，請嚴格審視其中的高風險元素，指出決策可能讓公司暴露於過度風險之處，並提出更謹慎的替代方案以確保長期收益。以下為交易員的決策：

{trader_decision}

你的任務是主動反駁『激進型』與『中立型』分析師的論點，凸顯他們可能忽略的潛在威脅或對永續性重視不足之處。請直接回應他們的要點，並運用下列資料來源，為低風險調整方案建立有說服力的論據：

市場研究報告: {market_research_report}
社群媒體情緒報告: {sentiment_report}
最新國際時事報告: {news_report}
公司基本面報告: {fundamentals_report}
目前對話紀錄: {history}
激進型分析師最後回應: {current_risky_response}
中立型分析師最後回應: {current_neutral_response}
若其他觀點尚未發言，請勿幻想內容，僅呈現你的觀點。

請透過質疑對方過度樂觀的假設並強調他們忽略的潛在下行風險，逐點回應對方論點，展示保守立場為公司資產提供最安全途徑的理由。聚焦於辯論與批判，凸顯低風險策略優於他們方案的優勢。以口語對話方式輸出，不需任何特殊格式。"""

        response = llm.invoke(prompt)

        argument = f"Safe Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": safe_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Safe",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return safe_node
