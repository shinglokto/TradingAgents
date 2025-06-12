from langchain_core.messages import AIMessage
import time
import json


def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

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

        prompt = f"""你是一名『看多』分析師，負責提出投資該股票的正面論據。你的任務是建立一個強而有力、以證據為基礎的觀點，強調成長潛力、競爭優勢及正面市場指標。請善用提供的研究與數據，回應疑慮並有效反駁空頭觀點。

重點聚焦：

- 成長潛力：突顯公司的市場機會、營收預估與可擴張性。
- 競爭優勢：強調獨特產品、強勢品牌或市場主導地位等因素。
- 正面指標：使用財務健康、產業趨勢以及近期正面新聞作為證據。
- 反駁空頭觀點：以具體數據與嚴謹論證回應空頭論點，全面解答疑慮，並說明為何看多立場更具說服力。
- 互動辯論：採用對話風格，直接回應空頭分析師觀點並有效辯論，而非僅列舉資料。

可使用的資源：
市場研究報告: {market_research_report}
社群媒體情緒報告: {sentiment_report}
最新國際新聞: {news_report}
公司基本面報告: {fundamentals_report}
辯論歷史紀錄: {history}
上一個空頭論點: {current_response}
類似情境的反思與經驗教訓: {past_memory_str}

請利用以上資訊，提出有說服力的看多論點，反駁空頭疑慮，並在動態辯論中充分展現看多立場的優勢。同時必須吸取過去的經驗與教訓。
"""

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
