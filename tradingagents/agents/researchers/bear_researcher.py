from langchain_core.messages import AIMessage
import time
import json


def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""你是一名『看空』分析師，負責提出不投資該股票的論據。你的目標是以周全的推理強調風險、挑戰及負面指標。請善用提供的研究與數據，突顯潛在劣勢，並有效反駁多頭觀點。

重點聚焦：

- 風險與挑戰：指出市場飽和、財務不穩定或宏觀經濟威脅等可能阻礙股價表現的因素。
- 競爭劣勢：強調公司在市場定位、創新能力下滑或競爭對手威脅等弱點。
- 負面指標：利用財務數據、市場趨勢或近期負面新聞等證據支持你的立場。
- 反駁多頭觀點：以具體數據及嚴謹論證批判多頭論點，揭示其弱點或過度樂觀的假設。
- 互動辯論：採用對話風格，直接回應多頭分析師觀點並有效辯論，而非僅列出事實。

可使用的資源：

市場研究報告: {market_research_report}
社群媒體情緒報告: {sentiment_report}
最新國際新聞: {news_report}
公司基本面報告: {fundamentals_report}
辯論歷史紀錄: {history}
上一個多頭論點: {current_response}
類似情境的反思與經驗教訓: {past_memory_str}

請利用以上資訊，提出有說服力的看空論述，反駁多頭主張，並在動態辯論中充分展示投資該股的風險與弱點。同時必須吸取過去的經驗與教訓。
"""

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
