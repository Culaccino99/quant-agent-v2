"""Agent 运行时上下文：通过 contextvars 透传用户信息和主事件循环"""
import asyncio
from contextvars import ContextVar

current_user_id: ContextVar[str] = ContextVar("current_user_id", default="")
main_event_loop: ContextVar[asyncio.AbstractEventLoop] = ContextVar("main_event_loop")


def run_on_main_loop(coro, timeout: float = 30):
    """在主事件循环上执行异步协程（供工作线程中的同步工具调用）"""
    loop = main_event_loop.get(None)
    if loop is None:
        raise RuntimeError("主事件循环未设置，无法执行异步操作")
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)
