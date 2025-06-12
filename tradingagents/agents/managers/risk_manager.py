import time
import json


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["news_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""作為風險管理法官兼辯論主持人，你的目標是評估三位風險分析師——激進（Risky）、中立（Neutral）、保守（Safe／Conservative）——的辯論，並為交易員決定最佳行動方案。你的決策必須給出明確建議：買入、賣出或持有。只有在有充分且具體論點支持時才選擇「持有」，不要因各方似乎皆有道理而退而求其次。務求清晰果斷。

決策指引：
1. **摘要關鍵論點**：擷取每位分析師最具說服力且與情境最相關的觀點。
2. **提供理由**：以辯論中的直接引述與反駁支持你的建議。
3. **優化交易員計劃**：以交易員原始計劃 **{trader_plan}** 為基礎，根據分析師見解進行調整。
4. **從過往錯誤學習**：運用 **{past_memory_str}** 的教訓，修正在先前判斷中的偏差，提升當前決策準確度，避免做出虧損的買／賣／持有指令。

成果交付：
- 明確且可執行的建議：買入、賣出或持有。
- 立基於辯論內容與過往反思的詳細推理。

---

**分析師辯論歷史：**  
{history}

---

專注於可執行的洞見與持續改進。汲取過往經驗，批判性地評估所有觀點，確保每項決策都能帶來更佳成果。"""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "risky_history": risk_debate_state["risky_history"],
            "safe_history": risk_debate_state["safe_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_risky_response": risk_debate_state["current_risky_response"],
            "current_safe_response": risk_debate_state["current_safe_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node
