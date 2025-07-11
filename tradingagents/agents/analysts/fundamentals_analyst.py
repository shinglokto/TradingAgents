from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_fundamentals_analyst(llm, toolkit):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [toolkit.get_fundamentals_openai]
        else:
            tools = [
                toolkit.get_finnhub_company_insider_sentiment,
                toolkit.get_finnhub_company_insider_transactions,
                toolkit.get_simfin_balance_sheet,
                toolkit.get_simfin_cashflow,
                toolkit.get_simfin_income_stmt,
            ]

        system_message = (
            "你是一位研究員，負責分析某公司過去一週的基本面資訊。請撰寫一份全面的報告，涵蓋該公司的財務文件、公司概況、基礎財務數據、財務歷史、內部人情緒以及內部人交易，以提供交易員完整視角。請盡可能詳盡，切勿僅以『趨勢好壞不一』帶過，務必提供細緻且具洞察力的分析，協助交易決策。"
            + " 請在報告結尾附上一個 Markdown 表格，將重點整理成易讀格式。",
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
                    " 供你參考，當前日期為 {current_date}。目標公司為 {ticker}",
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
            "fundamentals_report": result.content,
        }

    return fundamentals_analyst_node
