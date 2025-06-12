from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_news_analyst(llm, toolkit):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [toolkit.get_global_news_openai, toolkit.get_google_news]
        else:
            tools = [
                toolkit.get_finnhub_news,
                toolkit.get_reddit_news,
                toolkit.get_google_news,
            ]

        # 中文翻譯
        system_message = (
            "你是一名新聞研究員，負責分析過去一週的最新新聞與趨勢。請撰寫一份全面報告，說明與交易及宏觀經濟相關的全球現況。請綜合 EODHD 與 finnhub 的新聞來源以確保完整性。切勿僅以『趨勢好壞不一』帶過，務必提供細緻且具洞察力的分析，協助交易決策。"
            + " 請在報告結尾附上一個 Markdown 表格，將重點整理成易讀格式。"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一個樂於助人的 AI 助手，並會與其他助手協同工作。"
                    " 使用所提供的工具來推進問題的解答。"
                    " 若你無法完全回答，亦無妨；擁有不同工具的其他助手會接手你未完成的部分。"
                    " 執行你力所能及的操作以推進進度。"
                    " 若你或其他助手已提出最終交易建議：**BUY/HOLD/SELL** 或其他可交付成果，"
                    " 請在回覆前綴 FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** 讓團隊知道可以結束。"
                    " 你可以使用以下工具：{tool_names}.\n{system_message}"
                    " 供你參考，當前日期為 {current_date}。我們關注的公司為 {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        return {
            "messages": [result],
            "news_report": result.content,
        }

    return news_analyst_node
