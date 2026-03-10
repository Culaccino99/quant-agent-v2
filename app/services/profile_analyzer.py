"""用户画像分析器：基于对话内容，用 LLM 判断是否股票相关并提取用户投资偏好"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from loguru import logger

from app.agent.llm_client import get_deepseek_llm

_PROFILE_PROMPT_TEMPLATE = """\
# 任务
你是一个用户画像分析助手。根据下方的「用户消息」和「AI回复」，完成两件事：
1. 判断本次对话是否与 **股票投资 / A股市场 / 金融理财** 相关。
2. 如果相关，从对话中提取用户的投资偏好信息。

# 输出格式
严格输出一个 JSON 对象（不要输出任何多余文字），字段如下：

```json
{"is_stock_related": true/false, "investment_style": "conservative"|"balanced"|"aggressive"|null, "focus_sectors": ["行业1","行业2"]|null, "risk_tolerance": 1-10的整数|null}
```

## 字段说明
- `is_stock_related`：本次对话是否与股票/投资/金融相关。闲聊、技术问题、生活问题等返回 false。
- `investment_style`：用户投资风格。只能是以下三个值之一：
  - `conservative`（保守型）：偏好低风险、稳健收益、大盘蓝筹、高分红
  - `balanced`（平衡型）：风险收益兼顾、不极端
  - `aggressive`（进攻型）：偏好高弹性、追涨、小盘成长、题材炒作、短线交易
  - 如果无法判断，返回 null
- `focus_sectors`：用户关注的行业/板块列表，如 ["半导体", "新能源", "白酒"]。无法判断返回 null。
- `risk_tolerance`：风险承受度 1-10（1=极保守, 10=极激进）。无法判断返回 null。

# 少样本示例

## 示例 1：明确的股票讨论（进攻型）
用户消息：帮我看看精测电子300567，最近涨得猛，能不能追？
AI回复：精测电子近期受半导体设备国产化利好驱动上涨...建议关注回调机会...
输出：{"is_stock_related": true, "investment_style": "aggressive", "focus_sectors": ["半导体", "电子设备"], "risk_tolerance": 8}

## 示例 2：保守型投资者
用户消息：有没有分红比较稳定的股票推荐？我不想冒太大风险，就想长期持有吃分红。
AI回复：推荐关注长江电力、中国神华等高股息标的...
输出：{"is_stock_related": true, "investment_style": "conservative", "focus_sectors": ["电力", "煤炭", "高股息"], "risk_tolerance": 2}

## 示例 3：平衡型，多板块关注
用户消息：新能源和AI这两个方向你更看好哪个？
AI回复：新能源估值修复空间较大，AI方向弹性更强...
输出：{"is_stock_related": true, "investment_style": "balanced", "focus_sectors": ["新能源", "人工智能"], "risk_tolerance": 5}

## 示例 4：非股票相关
用户消息：今天天气怎么样？
AI回复：我是投研助手，天气方面我不太擅长...
输出：{"is_stock_related": false, "investment_style": null, "focus_sectors": null, "risk_tolerance": null}

## 示例 5：股票相关但偏好不明显
用户消息：贵州茅台最新财报怎么样？
AI回复：贵州茅台2025年营收同比增长15%...
输出：{"is_stock_related": true, "investment_style": null, "focus_sectors": ["白酒", "消费"], "risk_tolerance": null}

# 当前对话
用户消息：__USER_MESSAGE__
AI回复：__AGENT_REPLY__

请输出 JSON："""

_VALID_STYLES = {"conservative", "balanced", "aggressive"}

_llm = None


def _get_light_llm():
    """获取轻量 LLM 实例（低 temperature 保证输出稳定）"""
    global _llm
    if _llm is None:
        _llm = get_deepseek_llm(temperature=0)
        logger.info("[ProfileAnalyzer] LLM 实例已创建")
    return _llm


def analyze_user_profile(user_message: str, agent_reply: str) -> Optional[dict[str, Any]]:
    """
    分析对话内容，提取用户投资偏好。

    Returns:
        解析后的 dict，如 {"is_stock_related": True, "investment_style": "aggressive", ...}
        如果非股票相关或解析失败，返回 None。
    """
    start = time.time()
    logger.info("[ProfileAnalyzer] ▶ 开始分析用户画像")
    logger.info("[ProfileAnalyzer] 用户消息: {}", user_message[:200])

    prompt = (_PROFILE_PROMPT_TEMPLATE
              .replace("__USER_MESSAGE__", user_message)
              .replace("__AGENT_REPLY__", agent_reply[:1000]))

    try:
        llm = _get_light_llm()
        response = llm.invoke(prompt)
        raw_text = response.content if hasattr(response, "content") else str(response)
        logger.info("[ProfileAnalyzer] LLM 原始返回: {}", raw_text[:500])

        parsed = _extract_json(raw_text)
        if parsed is None:
            logger.warning("[ProfileAnalyzer] JSON 解析失败")
            return None

        is_related = parsed.get("is_stock_related", False)
        elapsed = time.time() - start
        logger.info("[ProfileAnalyzer] 分析完成: is_stock_related={}, 耗时={:.1f}s", is_related, elapsed)

        if not is_related:
            logger.info("[ProfileAnalyzer] 非股票相关对话, 跳过画像更新")
            return None

        result = _validate(parsed)
        logger.info("[ProfileAnalyzer] 提取结果: style={}, sectors={}, risk={}",
                    result.get("investment_style"),
                    result.get("focus_sectors"),
                    result.get("risk_tolerance"))
        return result

    except Exception as e:
        elapsed = time.time() - start
        logger.error("[ProfileAnalyzer] 分析失败(耗时{:.1f}s): {}", elapsed, e)
        return None


def _extract_json(text: str) -> Optional[dict]:
    """从 LLM 返回文本中提取 JSON 对象"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```") and not in_block:
                in_block = True
                continue
            if line.strip().startswith("```") and in_block:
                break
            if in_block:
                json_lines.append(line)
        text = "\n".join(json_lines)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _validate(parsed: dict) -> dict:
    """校验并清洗 LLM 返回的字段"""
    result: dict[str, Any] = {"is_stock_related": True}

    style = parsed.get("investment_style")
    if style in _VALID_STYLES:
        result["investment_style"] = style

    sectors = parsed.get("focus_sectors")
    if isinstance(sectors, list) and sectors:
        result["focus_sectors"] = [s for s in sectors if isinstance(s, str) and s.strip()]

    risk = parsed.get("risk_tolerance")
    if isinstance(risk, (int, float)) and 1 <= risk <= 10:
        result["risk_tolerance"] = int(risk)

    return result
