import time
import json


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_safe_response = risk_debate_state.get("current_safe_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""作為『中立型』風險分析師，你的角色是提供平衡視角，在評估交易員的決策或計畫時，同時衡量潛在收益與風險。你追求全面考量，上下兼顧正反面，並將整體市場趨勢、潛在經濟變動與多元化策略納入評估。以下為交易員的決策：

{trader_decision}

你的任務是同時挑戰『激進型』與『保守型』分析師，指出各自觀點中過度樂觀或過度謹慎之處。請運用以下資料來源的洞見，支持一個中度、可持續的調整策略：

市場研究報告: {market_research_report}
社群媒體情緒報告: {sentiment_report}
最新國際時事報告: {news_report}
公司基本面報告: {fundamentals_report}
目前對話紀錄: {history}
激進型分析師最後回應: {current_risky_response}
保守型分析師最後回應: {current_safe_response}
若其他觀點尚未發言，請勿幻想內容，僅呈現你的觀點。

請以批判式思維同時分析雙方，指出激進與保守論點的弱點，倡導更平衡的策略。逐一挑戰他們的觀點，說明中度風險策略如何兼顧成長潛力與避免極端波動。聚焦於辯論而非僅列舉資料，展示平衡視角能帶來最可靠結果。以口語對話方式輸出，不需任何特殊格式。"""

        response = llm.invoke(prompt)

        argument = f"Neutral Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": neutral_history + "\n" + argument,
            "latest_speaker": "Neutral",
            "current_risky_response": risk_debate_state.get(
                "current_risky_response", ""
            ),
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": argument,
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
