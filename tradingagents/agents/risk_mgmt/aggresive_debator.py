import time
import json


def create_risky_debator(llm):
    def risky_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        risky_history = risk_debate_state.get("risky_history", "")

        current_safe_response = risk_debate_state.get("current_safe_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""作為『激進型』風險分析師，你的角色是積極倡導高報酬、高風險的機會，並強調大膽策略與競爭優勢。當評估交易員的決策或計畫時，請集中火力於潛在上行空間、成長潛力與創新優勢——即使這些伴隨較高風險。運用提供的市場數據與情緒分析來鞏固你的論點，並挑戰相反觀點。請針對保守與中立分析師提出的每個要點，直接給出以數據為基礎的反駁與具說服力的推理，指出他們過於謹慎而錯失關鍵機會之處，或其假設過度保守之處。以下為交易員的決策：

{trader_decision}

你的任務是透過質疑與批判保守與中立立場，為交易員的決策打造一個有力的支持論證，展現高報酬觀點為何是最佳道路。請在論述中融入以下來源的洞見：

市場研究報告: {market_research_report}
社群媒體情緒報告: {sentiment_report}
最新國際時事報告: {news_report}
公司基本面報告: {fundamentals_report}
目前對話紀錄: {history}
保守分析師最後論點: {current_safe_response}
中立分析師最後論點: {current_neutral_response}
若其他觀點尚未發言，請勿幻想內容，僅呈現你的觀點。

請主動回應具體疑慮，駁斥對方論點中的邏輯弱點，並強調冒險所帶來的領先優勢。重點在於辯論與說服，而非僅列舉資料。逐一挑戰對方觀點，凸顯高風險策略為何最優。以對話口語方式輸出，不需任何特殊格式。"""

        response = llm.invoke(prompt)

        argument = f"Risky Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "risky_history": risky_history + "\n" + argument,
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Risky",
            "current_risky_response": argument,
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return risky_node
