from typing import Annotated, Sequence
from datetime import date, timedelta, datetime
from typing_extensions import TypedDict, Optional
from langchain_openai import ChatOpenAI
from tradingagents.agents import *
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph, START, MessagesState


# Researcher team state
class InvestDebateState(TypedDict):
    bull_history: Annotated[
        str, "多頭對話歷史"
    ]  # Bullish Conversation history
    bear_history: Annotated[
        str, "空頭對話歷史"
    ]  # Bearish Conversation history
    history: Annotated[str, "整體對話歷史"]  # Conversation history
    current_response: Annotated[str, "最新回覆"]  # Last response
    judge_decision: Annotated[str, "最終裁決"]  # Last response
    count: Annotated[int, "目前對話輪數"]  # Conversation length


# Risk management team state
class RiskDebateState(TypedDict):
    risky_history: Annotated[
        str, "激進型分析師對話歷史"
    ]  # Conversation history
    safe_history: Annotated[
        str, "保守型分析師對話歷史"
    ]  # Conversation history
    neutral_history: Annotated[
        str, "中立型分析師對話歷史"
    ]  # Conversation history
    history: Annotated[str, "整體對話歷史"]  # Conversation history
    latest_speaker: Annotated[str, "最後發言分析師"]
    current_risky_response: Annotated[
        str, "激進型分析師最新回覆"
    ]  # Last response
    current_safe_response: Annotated[
        str, "保守型分析師最新回覆"
    ]  # Last response
    current_neutral_response: Annotated[
        str, "中立型分析師最新回覆"
    ]  # Last response
    judge_decision: Annotated[str, "裁決者決定"]
    count: Annotated[int, "目前對話輪數"]  # Conversation length


class AgentState(MessagesState):
    company_of_interest: Annotated[str, "目標交易公司"]
    trade_date: Annotated[str, "交易日期"]

    sender: Annotated[str, "發送此訊息的代理人"]

    # research step
    market_report: Annotated[str, "市場分析師報告"]
    sentiment_report: Annotated[str, "社群媒體分析師報告"]
    news_report: Annotated[
        str, "新聞研究員對當前國際局勢之報告"
    ]
    fundamentals_report: Annotated[str, "基本面研究員報告"]

    # researcher team discussion step
    investment_debate_state: Annotated[
        InvestDebateState, "投資與否辯論之當前狀態"
    ]
    investment_plan: Annotated[str, "分析師產生之投資計畫"]

    trader_investment_plan: Annotated[str, "交易員產生之計畫"]

    # risk management team discussion step
    risk_debate_state: Annotated[
        RiskDebateState, "風險評估辯論之當前狀態"
    ]
    final_trade_decision: Annotated[str, "風險分析師所作之最終決策"]
