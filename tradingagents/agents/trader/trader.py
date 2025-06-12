import functools
import time
import json


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        # 將使用者提示翻譯為繁體中文
        context = {
            "role": "user",
            "content": f"根據團隊分析師的全面分析，以下為針對 {company_name} 制定的投資計畫。此計畫融合了當前技術面趨勢、宏觀經濟指標與社群媒體情緒。請以此計畫為基礎，評估你的下一步交易決策。\n\n建議投資計畫: {investment_plan}\n\n善用這些洞見，做出具策略性的明智決策。",
        }

        messages = [
            {
                "role": "system",
                "content": f"""你是一名交易代理人，負責分析市場數據以作出投資決策。根據你的分析，請給出明確建議：買入、賣出或持有。最後務必以『FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**』結尾，以確認你的建議。別忘了運用過去決策的經驗教訓，避免重蹈覆轍。以下為來自相似情境的反思與學到的教訓：{past_memory_str}""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
