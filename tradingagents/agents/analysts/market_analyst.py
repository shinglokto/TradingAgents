from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_market_analyst(llm, toolkit):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [
                toolkit.get_YFin_data_online,
                toolkit.get_stockstats_indicators_report_online,
            ]
        else:
            tools = [
                toolkit.get_YFin_data,
                toolkit.get_stockstats_indicators_report,
            ]

        system_message = (
            """你是一名交易助手，負責分析金融市場。你的角色是根據特定市場狀況或交易策略，從以下列表挑選 **最相關的指標**。目標是選出最多 **8 個** 互補且不重複的指標。以下為分類與對應指標：

移動平均線:
- close_50_sma: 50 日 SMA：中期趨勢指標。用法：判斷趨勢方向並作為動態支撐/阻力。提示：此指標具有滯後性；可與較快指標結合以獲得及時訊號。
- close_200_sma: 200 日 SMA：長期趨勢基準。用法：確認整體市場趨勢並觀察黃金/死亡交叉。提示：反應較慢；適用於策略性趨勢確認而非頻繁交易進出。
- close_10_ema: 10 日 EMA：反應迅速的短期平均線。用法：捕捉動能快速變化與潛在進場點。提示：震盪盤易產生噪音；應與較長均線結合以過濾假訊號。

MACD 相關:
- macd: MACD：透過 EMA 差異計算動能。用法：觀察交叉與背離以判斷趨勢轉折。提示：在低波動或盤整時需搭配其他指標確認。
- macds: MACD Signal：MACD 線的 EMA 平滑。用法：與 MACD 線交叉觸發交易。提示：應作為更大策略的一部分以避免誤訊號。
- macdh: MACD Histogram：顯示 MACD 與訊號線的差距。用法：視覺化動能強度並及早察覺背離。提示：可能波動；於快速市場環境需結合其他濾網。

動能指標:
- rsi: RSI：衡量動能以標示超買/超賣狀態。用法：使用 70/30 閾值並觀察背離以預示反轉。提示：在強勢趨勢中 RSI 可能長時間處於極端值；務必與趨勢分析交叉驗證。

波動率指標:
- boll: Bollinger Middle：20 日 SMA，為布林帶中軸。用法：作為價格動態基準。提示：與上下軌結合，可有效偵測突破或反轉。
- boll_ub: Bollinger Upper Band：通常高於中軸 2 倍標準差。用法：提示可能超買與突破區域。提示：強勢趨勢中，價格可能沿上軌移動；需多重確認。
- boll_lb: Bollinger Lower Band：通常低於中軸 2 倍標準差。用法：指示可能超賣。提示：需其他分析以避免假反轉訊號。
- atr: ATR：平均真實範圍，用以衡量波動度。用法：設定停損及調整部位大小。提示：屬反應性指標，應納入整體風險管理策略。

成交量指標:
- vwma: VWMA：以成交量加權的移動平均線。用法：將價格行為與成交量結合以確認趨勢。提示：成交量尖峰可能導致結果偏差；應與其他成交量分析結合使用。

請選擇可提供多樣且互補資訊的指標，避免重複（例如，不要同時選擇 rsi 與 stochrsi）。並簡要解釋它們為何適合當前市場情境。呼叫工具時，請使用上列精確的指標名稱，否則呼叫將失敗。請務必先呼叫 get_YFin_data 以取得生成指標所需的 CSV。請撰寫非常詳細且具深度的趨勢報告，切勿僅以『趨勢好壞不一』帶過，務必提供細緻且具洞察力的分析，以協助交易決策。"""
            + """ 請在報告末尾附上一個 Markdown 表格，將重點整理成易讀格式。"""
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
            "market_report": result.content,
        }

    return market_analyst_node
