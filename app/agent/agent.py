"""LangChain ReAct Agent：使用 DeepSeek + 工具链进行投研分析"""
from __future__ import annotations

import time
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents import create_agent
from loguru import logger

from app.agent.context import current_user_id
from app.agent.llm_client import get_deepseek_llm
from app.tools import (
    get_fundamental_data,
    search_news_rag,
    search_doc_rag,
    create_alert_rule,
    delete_alert_rule,
    list_alert_rules,
    toggle_alert_rule,
)

SYSTEM_PROMPT = """
# Role: A股资深智能投研与预警决策大脑

## 核心定位
你是一位拥有深厚A股市场经验的首席AI投研助理。你的任务是接收实时市场异动预警，自主调用外部数据工具进行深度归因分析，并输出符合**当前用户投资偏好**的投研简报。同时你也负责帮用户管理预警规则。

## 用户画像与分析视角
系统会额外提供一段"用户画像原始数据"（通常为 JSON），以及最近若干轮对话历史，你需要：
1. 先阅读这些画像数据和历史对话，根据其中的信息**自行推断/细化**用户的真实画像（如：风险偏好、偏好风格、关注板块等），而不是盲目套用固定风格。
2. 在分析过程中，根据你推断的画像自动调整视角与重点：
   - 对偏进攻型用户，可以多强调成长性、弹性、事件驱动机会；
   - 对偏保守型用户，应多关注现金流稳健、估值安全边际与回撤风险；
   - 对画像不明确的用户，可以按"平衡型"处理，但在必要时说明画像信息有限。
3. 如果你发现"存储画像"与"最近行为"明显不一致，可以在报告中简要指出这种偏差，并以你综合判断后的画像为准进行分析。

你不需要把内部推断过程全部展开说明，但在报告中应通过措辞与侧重点体现出对该用户画像的理解。

## 工具使用规范

### 投研分析工具
1. **`search_news_rag` (查资讯与研报)：** 用于查询特定股票的最新突发新闻、券商研报摘要、产业链动态及市场舆情。当收到异动预警时，优先调用此工具寻找异动的"直接导火索"或"情绪催化剂"。
2. **`get_fundamental_data` (查基本面与估值)：** 用于获取公司的核心财务指标、估值（如 PE/PB）、营收与净利润增长率等结构化数据。用于验证该公司当前的业绩底子是否能支撑其异动逻辑。
3. **`search_doc_rag` (查用户上传文档)：** 用于在用户上传的文档库中检索相关内容。当用户提问涉及其上传的研报、分析报告、资料文档时，调用此工具获取文档中的相关段落。

### 预警管理工具
4. **`create_alert_rule` (创建预警)：** 当用户说"帮我设一个xxx的预警"、"监控xxx涨幅超过x%通知我"时调用。需要参数：stock_code(6位代码)、rule_type(price_up/price_down/turnover)、threshold(阈值数字)。
5. **`list_alert_rules` (查看预警)：** 当用户说"我有哪些预警"、"看看我的预警规则"时调用。
6. **`delete_alert_rule` (删除预警)：** 当用户说"删除规则X"、"取消预警X"时调用。需要参数：rule_id。
7. **`toggle_alert_rule` (暂停/恢复预警)：** 当用户说"暂停规则X"或"恢复规则X"时调用。需要参数：rule_id 和 action(pause/resume)。

### 执行逻辑
- 投研分析：通常先用 `search_news_rag` 明确事件，再用 `get_fundamental_data` 验证质地。如果用户提到了其上传的文档或报告内容，使用 `search_doc_rag` 检索。
- 预警管理：用户表达想要设置/查看/删除预警时，直接调用对应工具，并将操作结果清晰告知用户。
- 如果工具返回的数据为空或不足以得出确切结论，请在报告中如实说明"当前数据不足"，绝对不可自行捏造事实。

## 输出格式要求

### 投研分析报告 (Feishu Markdown)
当进行投研分析时，输出以下格式，总字数严控在 250 字以内：

**【用户画像理解】**
(用一到两句话总结你对当前用户风险偏好、风格倾向和关注方向的综合判断。)

**【异动归因总结】**
(用一到两句话，一针见血地概括导致本次异动或预警的核心驱动事件。)

**【数据与逻辑印证】**
- **事件催化：** (精准提炼 `search_news_rag` 获取的关键利好/利空事实)
- **基本面成色：** (列出 `get_fundamental_data` 查到的关键盈利能力、估值水位或增速数据，简评优劣)

**【策略研判】**
(结合用户画像给出清晰的操作建议或风险提示。)

### 预警管理回复
当进行预警规则管理时，直接用简洁友好的语言告知操作结果即可，无需使用上述报告格式。
"""

_TOOLS = [
    get_fundamental_data,
    search_news_rag,
    search_doc_rag,
    create_alert_rule,
    list_alert_rules,
    delete_alert_rule,
    toggle_alert_rule,
]

_agent_graph = None


def _get_agent():
    global _agent_graph
    if _agent_graph is None:
        logger.info("[Agent] 初始化 ReAct Agent (DeepSeek + {} 个工具)", len(_TOOLS))
        llm = get_deepseek_llm()
        _agent_graph = create_agent(
            model=llm,
            tools=_TOOLS,
            system_prompt=SYSTEM_PROMPT,
        )
        logger.info("[Agent] Agent 初始化完成")
    return _agent_graph


def _truncate(text: str, max_len: int = 300) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


def run_agent(
    query: str,
    user_profile: Optional[str] = None,
    chat_history: Optional[list[dict[str, Any]]] = None,
    feishu_open_id: Optional[str] = None,
) -> str:
    start_time = time.time()
    logger.info("[Agent] 用户提问: {}", query)

    # 设置当前用户上下文（供预警管理工具读取）
    if feishu_open_id:
        current_user_id.set(feishu_open_id)

    # 组装上下文
    messages: list[Any] = []

    if user_profile:
        logger.info("[Agent] 注入用户画像: {}", _truncate(user_profile))
        messages.append(
            SystemMessage(content=(
                "以下是当前用户的投资画像原始数据(JSON)，请先据此结合历史行为自行推断出更精细的用户画像，"
                "再基于你的推断来解读本次问题：\n"
                f"{user_profile}"
            ))
        )

    if chat_history:
        for i, h in enumerate(chat_history):
            role = h.get("role", "")
            content = h.get("content", "")
            if not content:
                continue
            logger.info("[Agent]   历史[{}] role={}, content={}", i, role, _truncate(content, 100))
            if role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))

    messages.append(HumanMessage(content=query))
    logger.info("[Agent] 消息列表组装完成: 共 {} 条 ({})",
                len(messages),
                ", ".join(type(m).__name__ for m in messages))

    # 调用 Agent
    agent = _get_agent()
    result = agent.invoke({"messages": messages})
    out_messages = result.get("messages", [])

    # 逐步打印 Agent 执行过程
    logger.info("[Agent] Agent 执行完毕, 共产生 {} 条消息:", len(out_messages))
    for i, msg in enumerate(out_messages):
        msg_type = getattr(msg, "type", type(msg).__name__)
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            content = str(content)

        if msg_type == "human":
            logger.info("[Agent]   [{}] 👤 Human: {}", i, _truncate(content, 200))
        elif msg_type == "ai":
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    logger.info("[Agent]   [{}] 🤖 AI → 调用工具: {}({})",
                                i, tc.get("name", "?"), _truncate(str(tc.get("args", {})), 200))
            elif content:
                logger.info("[Agent]   [{}] 🤖 AI 回复: {}", i, _truncate(content, 500))
            else:
                logger.info("[Agent]   [{}] 🤖 AI (空内容)", i)
        elif msg_type == "tool":
            tool_name = getattr(msg, "name", "?")
            logger.info("[Agent]   [{}] 🔧 Tool[{}] 返回: {}", i, tool_name, _truncate(content, 500))
        elif msg_type == "system":
            logger.info("[Agent]   [{}] 📋 System: {}", i, _truncate(content, 200))
        else:
            logger.info("[Agent]   [{}] {} : {}", i, msg_type, _truncate(content, 200))

    # 提取最终回答
    if not out_messages:
        logger.warning("[Agent] ⚠ 无消息返回")
        return "（无回复）"

    for msg in reversed(out_messages):
        if hasattr(msg, "content") and msg.content and getattr(msg, "type", "") == "ai":
            out = msg.content if isinstance(msg.content, str) else str(msg.content)
            elapsed = time.time() - start_time
            logger.info("[Agent] ✅ 最终输出: length={}, 耗时={:.1f}s", len(out), elapsed)
            logger.info("[Agent] 输出预览: {}", _truncate(out, 500))
            logger.info("=" * 60)
            return out

    elapsed = time.time() - start_time
    logger.warning("[Agent] ⚠ 未找到 AI 回复消息, 耗时={:.1f}s", elapsed)
    logger.info("=" * 60)
    return "（无回复）"
