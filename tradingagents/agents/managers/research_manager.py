import time
import json


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""作為投資組合經理兼辯論主持人，你的職責是嚴格審視本輪辯論，並作出明確決策：支持熊派分析師、牛派分析師，或僅在論點充分且有力時才選擇「持有」。

請簡明扼要地總結雙方的關鍵觀點，聚焦最具說服力的證據或推理。你的推薦——買入、賣出或持有——必須清晰且可執行。避免因雙方皆有道理而預設為「持有」；請以辯論中最強的論據為依據採取立場。

此外，為交易員制定一份詳細的投資計劃，內容須包含：
• 推薦：以最有力論點支撐的明確立場。
• 理由：說明這些論點如何導致你的結論。
• 策略行動：落實推薦的具體步驟。

請考慮你在類似情況下的過往錯誤，運用這些洞見精進決策，確保持續學習與改進。以自然對話方式呈現分析，勿使用特殊格式。

以下為你過往對錯誤的反思：
\"{past_memory_str}\"

以下為辯論內容：
辯論歷史：
{history}"""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
