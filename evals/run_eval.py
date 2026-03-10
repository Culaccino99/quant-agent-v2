#!/usr/bin/env python3
"""
离线评测脚本：对 Agent 进行多维度自动化评测

评测维度：
  1. 工具路由准确率 (Tool Routing Accuracy)  — Agent 是否调对了工具
  2. 参数解析准确率 (Arg Parsing Accuracy)   — 关键参数是否正确提取
  3. 回答质量评分   (Answer Quality, LLM-as-Judge)  — 由 LLM 对回答打分
  4. 拒绝率         (Rejection Rate)          — 非投研问题是否避免了工具调用
  5. 延迟统计       (Latency)                 — 每条用例的响应时间

使用方式：
  cd quant-agent-v2
  python -m evals.run_eval                     # 默认使用 evals/test_cases.json
  python -m evals.run_eval --cases my.json     # 指定用例文件
  python -m evals.run_eval --output report.json # 指定输出文件
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from loguru import logger

# ---------------------------------------------------------------------------
# LangSmith 评测集成（可选）
# ---------------------------------------------------------------------------
_USE_LANGSMITH = (
    os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    and os.environ.get("LANGCHAIN_API_KEY", "")
)


def _init_langsmith_dataset(test_cases: list[dict]) -> Any:
    """如果 LangSmith 可用，创建/更新评测数据集并返回 dataset 对象"""
    if not _USE_LANGSMITH:
        return None
    try:
        from langsmith import Client
        client = Client()
        ds_name = "quant-agent-eval"
        dataset = client.create_dataset(ds_name, description="A股投研 Agent 离线评测集")
        for tc in test_cases:
            client.create_example(
                inputs={"query": tc["query"]},
                outputs={
                    "expected_tools": tc.get("expected_tools", []),
                    "expected_args": tc.get("expected_args", {}),
                },
                dataset_id=dataset.id,
                metadata={"id": tc["id"], "category": tc["category"]},
            )
        logger.info("[Eval] LangSmith 数据集已创建: {}", ds_name)
        return dataset
    except Exception as e:
        logger.warning("[Eval] LangSmith 数据集创建失败(不影响本地评测): {}", e)
        return None


# ---------------------------------------------------------------------------
# 核心评测：调用 Agent 并解析中间步骤
# ---------------------------------------------------------------------------
def _build_agent():
    """构建一个和线上相同的 ReAct Agent，但不走飞书"""
    from app.agent.llm_client import get_deepseek_llm
    from app.agent.agent import SYSTEM_PROMPT, _TOOLS
    from langchain.agents import create_agent

    llm = get_deepseek_llm()
    return create_agent(model=llm, tools=_TOOLS, system_prompt=SYSTEM_PROMPT)


def _run_single_case(agent, tc: dict) -> dict:
    """执行单条测试用例，返回结构化评测结果"""
    from langchain_core.messages import HumanMessage

    query = tc["query"]
    case_id = tc["id"]
    expected_tools = set(tc.get("expected_tools", []))
    expected_args = tc.get("expected_args", {})

    logger.info("[Eval] ▶ {} | {} | {}", case_id, tc["category"], query)

    t0 = time.time()
    try:
        result = agent.invoke({"messages": [HumanMessage(content=query)]})
    except Exception as e:
        logger.error("[Eval] {} 执行异常: {}", case_id, e)
        return {
            "id": case_id,
            "category": tc["category"],
            "query": query,
            "error": str(e),
            "pass": False,
        }
    latency = time.time() - t0

    messages = result.get("messages", [])

    # 提取 Agent 实际调用的工具
    actual_tools: list[str] = []
    actual_args: dict[str, Any] = {}
    final_answer = ""

    for msg in messages:
        msg_type = getattr(msg, "type", "")
        if msg_type == "ai":
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc_call in tool_calls:
                    name = tc_call.get("name", "")
                    args = tc_call.get("args", {})
                    actual_tools.append(name)
                    actual_args.update(args)
            elif getattr(msg, "content", ""):
                final_answer = msg.content if isinstance(msg.content, str) else str(msg.content)

    actual_tools_set = set(actual_tools)

    # --- 指标 1: 工具路由准确率 ---
    if expected_tools:
        tool_match = expected_tools.issubset(actual_tools_set)
    else:
        tool_match = len(actual_tools_set) == 0

    # --- 指标 2: 关键参数匹配 ---
    arg_match_details: dict[str, bool] = {}
    for key, val in expected_args.items():
        actual_val = actual_args.get(key)
        if actual_val is not None:
            arg_match_details[key] = str(actual_val) == str(val)
        else:
            arg_match_details[key] = False
    arg_match = all(arg_match_details.values()) if arg_match_details else True

    record = {
        "id": case_id,
        "category": tc["category"],
        "query": query,
        "description": tc.get("description", ""),
        "expected_tools": sorted(expected_tools),
        "actual_tools": sorted(actual_tools_set),
        "tool_routing_pass": tool_match,
        "expected_args": expected_args,
        "actual_args": {k: actual_args.get(k) for k in expected_args} if expected_args else {},
        "arg_parsing_pass": arg_match,
        "arg_details": arg_match_details,
        "final_answer": final_answer[:500],
        "final_answer_length": len(final_answer),
        "latency_s": round(latency, 2),
        "error": None,
    }

    status = "✅" if (tool_match and arg_match) else "❌"
    logger.info("[Eval] {} {} | tools={} args={} | {:.1f}s",
                status, case_id, tool_match, arg_match, latency)

    return record


# ---------------------------------------------------------------------------
# LLM-as-Judge 回答质量评分
# ---------------------------------------------------------------------------
_JUDGE_PROMPT = """\
你是一个严格的评测专家，负责评估 AI 投研助理的回答质量。

**用户提问：**
{query}

**AI 回答：**
{answer}

请从以下维度打分（每项 1-5 分），并给出一句话理由：
1. **相关性**：回答是否切题
2. **准确性**：事实是否正确，有无捏造
3. **完整性**：是否覆盖了用户关心的要点
4. **可操作性**：是否给出了有用的分析或建议

请严格输出 JSON 格式：
{{"relevance": 分数, "accuracy": 分数, "completeness": 分数, "actionability": 分数, "reason": "一句话理由"}}"""


def _judge_answer(llm, query: str, answer: str) -> dict:
    """用 LLM 对单条回答做质量评分"""
    if not answer or len(answer) < 10:
        return {"relevance": 1, "accuracy": 1, "completeness": 1, "actionability": 1,
                "reason": "回答过短或为空", "avg_score": 1.0}
    try:
        prompt = _JUDGE_PROMPT.format(query=query, answer=answer[:2000])
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        text = text.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "", 1).strip()
        scores = json.loads(text)
        avg = round(sum(scores.get(k, 1) for k in ["relevance", "accuracy", "completeness", "actionability"]) / 4, 2)
        scores["avg_score"] = avg
        return scores
    except Exception as e:
        logger.warning("[Judge] 评分解析失败: {}", e)
        return {"relevance": 0, "accuracy": 0, "completeness": 0, "actionability": 0,
                "reason": f"评分失败: {e}", "avg_score": 0}


# ---------------------------------------------------------------------------
# 生成汇总报告
# ---------------------------------------------------------------------------
def _generate_report(records: list[dict], quality_scores: list[dict]) -> dict:
    total = len(records)
    errors = sum(1 for r in records if r.get("error"))

    tool_pass = sum(1 for r in records if r.get("tool_routing_pass"))
    arg_pass = sum(1 for r in records if r.get("arg_parsing_pass"))
    valid = total - errors

    # 分类别统计
    categories: dict[str, dict] = {}
    for r in records:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "tool_pass": 0, "arg_pass": 0, "errors": 0}
        categories[cat]["total"] += 1
        if r.get("error"):
            categories[cat]["errors"] += 1
        if r.get("tool_routing_pass"):
            categories[cat]["tool_pass"] += 1
        if r.get("arg_parsing_pass"):
            categories[cat]["arg_pass"] += 1

    latencies = [r["latency_s"] for r in records if r.get("latency_s") is not None]
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0

    avg_quality = 0
    if quality_scores:
        avg_quality = round(sum(q.get("avg_score", 0) for q in quality_scores) / len(quality_scores), 2)

    report = {
        "summary": {
            "total_cases": total,
            "errors": errors,
            "tool_routing_accuracy": f"{tool_pass}/{valid} ({round(tool_pass / valid * 100, 1) if valid else 0}%)",
            "arg_parsing_accuracy": f"{arg_pass}/{valid} ({round(arg_pass / valid * 100, 1) if valid else 0}%)",
            "avg_quality_score": f"{avg_quality}/5.0",
            "avg_latency_s": avg_latency,
            "p95_latency_s": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 2),
        },
        "by_category": {
            cat: {
                "total": info["total"],
                "tool_pass": info["tool_pass"],
                "arg_pass": info["arg_pass"],
                "errors": info["errors"],
                "tool_accuracy": f"{round(info['tool_pass'] / max(info['total'] - info['errors'], 1) * 100, 1)}%",
            }
            for cat, info in sorted(categories.items())
        },
        "details": records,
        "quality_scores": quality_scores,
    }
    return report


def _print_summary(report: dict):
    s = report["summary"]
    print("\n" + "=" * 60)
    print("  A股投研 Agent 离线评测报告")
    print("=" * 60)
    print(f"  用例总数:        {s['total_cases']}")
    print(f"  执行异常:        {s['errors']}")
    print(f"  工具路由准确率:  {s['tool_routing_accuracy']}")
    print(f"  参数解析准确率:  {s['arg_parsing_accuracy']}")
    print(f"  回答质量均分:    {s['avg_quality_score']}")
    print(f"  平均延迟:        {s['avg_latency_s']}s")
    print(f"  P95 延迟:        {s['p95_latency_s']}s")
    print("-" * 60)
    print("  分类别统计:")
    for cat, info in report["by_category"].items():
        print(f"    {cat:20s} | 总数={info['total']:2d} | 工具={info['tool_accuracy']:>6s} | 异常={info['errors']}")
    print("-" * 60)

    failed = [r for r in report["details"] if not r.get("tool_routing_pass") or not r.get("arg_parsing_pass")]
    if failed:
        print("  失败用例:")
        for r in failed:
            print(f"    {r['id']} | 工具={r.get('tool_routing_pass')} 参数={r.get('arg_parsing_pass')} | {r['query']}")
            if not r.get("tool_routing_pass"):
                print(f"      期望工具: {r.get('expected_tools')}")
                print(f"      实际工具: {r.get('actual_tools')}")
            if not r.get("arg_parsing_pass"):
                print(f"      参数明细: {r.get('arg_details')}")
    else:
        print("  🎉 所有用例全部通过！")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# LangSmith 评测运行器（可选）
# ---------------------------------------------------------------------------
def _run_langsmith_eval(agent, test_cases: list[dict], dataset):
    """如果启用 LangSmith，通过 langsmith 的 evaluate 接口运行"""
    if not _USE_LANGSMITH or dataset is None:
        return

    try:
        from langsmith import evaluate
        from langchain_core.messages import HumanMessage

        def predict(inputs: dict) -> dict:
            result = agent.invoke({"messages": [HumanMessage(content=inputs["query"])]})
            messages = result.get("messages", [])
            tools_called = []
            answer = ""
            for msg in messages:
                if getattr(msg, "type", "") == "ai":
                    for tc in getattr(msg, "tool_calls", []) or []:
                        tools_called.append(tc.get("name", ""))
                    if getattr(msg, "content", ""):
                        answer = msg.content
            return {"answer": answer, "tools_called": tools_called}

        def tool_routing_evaluator(run, example) -> dict:
            expected = set(example.outputs.get("expected_tools", []))
            actual = set(run.outputs.get("tools_called", []))
            if expected:
                score = 1.0 if expected.issubset(actual) else 0.0
            else:
                score = 1.0 if len(actual) == 0 else 0.0
            return {"key": "tool_routing", "score": score}

        results = evaluate(
            predict,
            data=dataset.name,
            evaluators=[tool_routing_evaluator],
            experiment_prefix="quant-agent-eval",
        )
        logger.info("[Eval] LangSmith 评测完成，可在 LangSmith Dashboard 查看结果")
    except Exception as e:
        logger.warning("[Eval] LangSmith 评测运行失败: {}", e)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="A股投研 Agent 离线评测")
    parser.add_argument("--cases", default=str(Path(__file__).parent / "test_cases.json"),
                        help="测试用例 JSON 文件路径")
    parser.add_argument("--output", default=str(Path(__file__).parent / "report.json"),
                        help="评测报告输出路径")
    parser.add_argument("--skip-judge", action="store_true",
                        help="跳过 LLM-as-Judge 质量评分（节省 API 调用）")
    parser.add_argument("--ids", nargs="*", default=None,
                        help="只运行指定 ID 的用例，如 --ids TC001 TC005")
    args = parser.parse_args()

    with open(args.cases, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    if args.ids:
        test_cases = [tc for tc in test_cases if tc["id"] in args.ids]
        logger.info("[Eval] 过滤后运行 {} 条用例: {}", len(test_cases), args.ids)

    logger.info("[Eval] 加载 {} 条测试用例", len(test_cases))
    logger.info("[Eval] LangSmith: {}", "已启用" if _USE_LANGSMITH else "未启用")

    # LangSmith 数据集
    ls_dataset = _init_langsmith_dataset(test_cases) if _USE_LANGSMITH else None

    # 构建 Agent
    logger.info("[Eval] 构建 Agent...")
    agent = _build_agent()

    # 逐条执行
    records: list[dict] = []
    for tc in test_cases:
        record = _run_single_case(agent, tc)
        records.append(record)

    # LLM-as-Judge
    quality_scores: list[dict] = []
    if not args.skip_judge:
        logger.info("[Eval] 启动 LLM-as-Judge 质量评分...")
        from app.agent.llm_client import get_deepseek_llm
        judge_llm = get_deepseek_llm(temperature=0)

        for rec in records:
            if rec.get("error"):
                quality_scores.append({"avg_score": 0, "reason": "执行异常"})
                continue
            answer = ""
            # 重新获取 final_answer（从 record 中无法取到完整回答，需要记录）
            # 这里用 final_answer_length 判断是否有回答
            if rec.get("final_answer_length", 0) > 10:
                score = _judge_answer(judge_llm, rec["query"], rec.get("final_answer", ""))
            else:
                score = {"avg_score": 2, "reason": "回答过短"}
            quality_scores.append(score)
    else:
        logger.info("[Eval] 跳过 LLM-as-Judge")

    # LangSmith 评测
    if _USE_LANGSMITH and ls_dataset:
        _run_langsmith_eval(agent, test_cases, ls_dataset)

    # 生成报告
    report = _generate_report(records, quality_scores)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("[Eval] 报告已保存: {}", args.output)

    _print_summary(report)


if __name__ == "__main__":
    main()
