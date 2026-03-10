"""飞书 Webhook 路由：消息接收 → 文本走 Agent 分析 / 文件走 RAG 入库"""
from __future__ import annotations

import asyncio
import json
import time
import traceback

from fastapi import APIRouter, BackgroundTasks, Request
from loguru import logger

from app.agent.agent import run_agent
from app.services import feishu_client
from app.services.feishu_client import build_analysis_card
from app.services.profile_service import get_or_create_user, get_user_profile_json, update_user_profile
from app.services.session_service import append_message, get_chat_history

router = APIRouter(tags=["feishu"])

_processed_event_ids: set[str] = set()


@router.post("/webhook")
async def feishu_webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()

    if "challenge" in body:
        logger.info("[Webhook] 飞书 URL Challenge 验证")
        return {"challenge": body["challenge"]}

    event_id = body.get("header", {}).get("event_id", "")
    if event_id in _processed_event_ids:
        logger.info("[Webhook] 重复事件跳过: {}", event_id)
        return {"code": 0}
    _processed_event_ids.add(event_id)
    if len(_processed_event_ids) > 10000:
        _processed_event_ids.clear()

    event = body.get("event", {})
    event_type = body.get("header", {}).get("event_type", "")

    if event_type == "im.message.receive_v1":
        message = event.get("message", {})
        msg_type = message.get("message_type", "")
        sender_id = event.get("sender", {}).get("sender_id", {}).get("open_id", "")
        message_id = message.get("message_id", "")
        chat_id = message.get("chat_id", "")
        chat_type = message.get("chat_type", "")

        if msg_type == "text":
            content = json.loads(message.get("content", "{}"))
            text = content.get("text", "").strip()
            background_tasks.add_task(
                _analyze_and_reply,
                sender_id=sender_id, text=text, message_id=message_id,
                chat_id=chat_id, chat_type=chat_type,
            )

        elif msg_type == "file":
            content = json.loads(message.get("content", "{}"))
            file_key = content.get("file_key", "")
            file_name = content.get("file_name", "unknown")
            logger.info("[Webhook] 文件消息: name={}, key={}", file_name, file_key)
            background_tasks.add_task(
                _process_file_upload,
                sender_id=sender_id, message_id=message_id,
                file_key=file_key, file_name=file_name,
                chat_id=chat_id, chat_type=chat_type,
            )

        else:
            logger.info("[Webhook] 暂不处理的消息类型: {}", msg_type)

    return {"code": 0}


async def _analyze_and_reply(
    sender_id: str, text: str, message_id: str, chat_id: str, chat_type: str,
):
    """后台任务：注入记忆 → Agent 分析 → 保存对话 → 推送飞书卡片"""
    task_start = time.time()

    # 把主事件循环存入 contextvars，供工作线程中的工具调度异步操作回主循环
    from app.agent.context import main_event_loop
    main_event_loop.set(asyncio.get_running_loop())

    try:
        # Step 1: 确保用户存在
        await get_or_create_user(sender_id)

        # Step 2: 读取用户画像
        user_profile = await get_user_profile_json(sender_id)

        # Step 3: 读取对话历史
        chat_history = await get_chat_history(sender_id)

        # Step 4: 保存用户消息
        await append_message(sender_id, "user", text)

        # Step 5: 调用 Agent
        agent_start = time.time()
        report = await asyncio.to_thread(
            run_agent, query=text, user_profile=user_profile,
            chat_history=chat_history, feishu_open_id=sender_id,
        )
        agent_elapsed = time.time() - agent_start
        logger.info("[Task] Agent 完成, 耗时={:.1f}s, 输出长度={}", agent_elapsed, len(report))

        # Step 6: 保存 Agent 回复
        await append_message(sender_id, "assistant", report)

        # Step 7: 构建飞书卡片并推送
        card = build_analysis_card(title="📊 AI 投研分析", content_md=report)
        if chat_type == "group":
            await feishu_client.send_card_message(receive_id=chat_id, card=card, receive_id_type="chat_id")
        else:
            await feishu_client.send_card_message(receive_id=sender_id, card=card, receive_id_type="open_id")

        # Step 8: 异步分析用户画像（不阻塞主流程，失败只打日志）
        try:
            from app.services.profile_analyzer import analyze_user_profile
            logger.info("[Task] Step8 分析用户画像...")
            profile_start = time.time()
            profile_result = await asyncio.to_thread(
                analyze_user_profile, user_message=text, agent_reply=report,
            )
            if profile_result and profile_result.get("is_stock_related"):
                await update_user_profile(
                    feishu_open_id=sender_id,
                    investment_style=profile_result.get("investment_style"),
                    focus_sectors=profile_result.get("focus_sectors"),
                    risk_tolerance=profile_result.get("risk_tolerance"),
                    merge_sectors=True,
                )
                logger.info("[Task] Step8 画像已更新, 耗时={:.1f}s", time.time() - profile_start)
            else:
                logger.info("[Task] Step8 非股票相关, 跳过画像更新, 耗时={:.1f}s", time.time() - profile_start)
        except Exception as profile_err:
            logger.warning("[Task] Step8 画像分析失败(不影响主流程): {}", profile_err)

        total_elapsed = time.time() - task_start
        logger.info("[Task] ✅ 文本消息处理完成: user={}, 总耗时={:.1f}s", sender_id, total_elapsed)

    except Exception as e:
        total_elapsed = time.time() - task_start
        logger.error("[Task] ❌ 处理失败(耗时{:.1f}s): {}\n{}", total_elapsed, e, traceback.format_exc())
        try:
            await feishu_client.reply_to_message(message_id, f"分析过程出错，请稍后重试: {e}")
        except Exception:
            logger.error("[Task] 回复错误消息也失败了")


async def _process_file_upload(
    sender_id: str, message_id: str, file_key: str, file_name: str,
    chat_id: str, chat_type: str,
):
    """后台任务：下载飞书文件 → 解析 → 切片 → Embedding 入 Qdrant"""
    task_start = time.time()
    logger.info("[FileTask] ▶ 开始处理文件上传: user={}, file={}", sender_id, file_name)

    try:
        await feishu_client.reply_to_message(message_id, f"📄 正在解析文件「{file_name}」，请稍候...")
        logger.info("[FileTask] Step1 已通知用户正在处理")

        # Step 2: 下载
        logger.info("[FileTask] Step2 下载文件...")
        dl_start = time.time()
        file_bytes = await feishu_client.download_file(message_id, file_key)
        logger.info("[FileTask] Step2 下载完成: size={}KB, 耗时={:.1f}s",
                    len(file_bytes) // 1024, time.time() - dl_start)

        # Step 3: 解析
        logger.info("[FileTask] Step3 解析文件...")
        from app.services.doc_parser import parse_file
        parse_start = time.time()
        chunks = await asyncio.to_thread(parse_file, file_bytes, file_name)
        logger.info("[FileTask] Step3 解析完成: {} 个切片, 耗时={:.1f}s",
                    len(chunks), time.time() - parse_start)

        if not chunks:
            await feishu_client.reply_to_message(
                message_id, f"⚠️ 文件「{file_name}」解析结果为空，可能是不支持的格式或内容为空。"
            )
            return

        if chunks:
            logger.info("[FileTask] 切片预览[0]: {}", chunks[0][:200])

        # Step 4: Embedding + 入库
        logger.info("[FileTask] Step4 Embedding 并写入 Qdrant...")
        from app.services.vector_store import add_doc_chunks
        embed_start = time.time()
        count = await asyncio.to_thread(add_doc_chunks, chunks, file_name, sender_id)
        logger.info("[FileTask] Step4 入库完成: {} 个向量, 耗时={:.1f}s",
                    count, time.time() - embed_start)

        # Step 5: 回复
        reply_text = (
            f"✅ 文件「{file_name}」已入库！\n"
            f"共解析为 {count} 个文本切片，已存入向量库。\n"
            f"你现在可以针对这份文档提问，例如：「这份研报的核心观点是什么？」"
        )
        await feishu_client.reply_to_message(message_id, reply_text)

        total_elapsed = time.time() - task_start
        logger.info("[FileTask] ✅ 文件处理完成: file={}, chunks={}, 总耗时={:.1f}s",
                    file_name, count, total_elapsed)

    except Exception as e:
        total_elapsed = time.time() - task_start
        logger.error("[FileTask] ❌ 文件处理失败(耗时{:.1f}s): {}\n{}",
                     total_elapsed, e, traceback.format_exc())
        try:
            await feishu_client.reply_to_message(message_id, f"❌ 文件处理失败: {e}")
        except Exception:
            logger.error("[FileTask] 回复错误消息也失败了")
