# -*- coding: utf-8 -*-
"""
Tool: 基本面查询。多数据源 + 限频 + 降级：三路数据源各尝试一次，失败则切换下一路。
"""
import os

for key in list(os.environ):
    if "proxy" in key.lower() and key != "GOPROXY":
        os.environ.pop(key, None)

import urllib.request
urllib.request.getproxies = lambda: {}
urllib.request.getproxies_environment = lambda: {}

import time

import akshare as ak
import requests
from langchain_core.tools import tool
from loguru import logger

_MIN_INTERVAL_SEC = 2
_FALLBACK_DELAY_SEC = 2
_last_request_time = 0.0

_EM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "application/json",
    "Connection": "keep-alive",
}
_CODE_MAP = {
    "f57": "股票代码", "f58": "股票简称", "f84": "总股本", "f85": "流通股",
    "f127": "行业", "f116": "总市值", "f117": "流通市值", "f189": "上市时间", "f43": "最新",
}


def _rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_INTERVAL_SEC and _last_request_time > 0:
        wait = _MIN_INTERVAL_SEC - elapsed
        logger.info("[Tool:fundamental] 限频等待 {:.1f}s", wait)
        time.sleep(wait)
    _last_request_time = time.time()


def _fetch_source1_direct_em(symbol: str, timeout: float = 10) -> dict:
    """数据源 1：直接请求东方财富单股接口"""
    logger.info("[Tool:fundamental] 尝试数据源1(东方财富直连): {}", symbol)
    market_code = 1 if symbol.startswith("6") else 0
    params = {
        "fltt": "2", "invt": "2",
        "fields": "f43,f57,f58,f84,f85,f116,f117,f127,f189",
        "secid": f"{market_code}.{symbol}",
    }
    session = requests.Session()
    session.trust_env = False
    session.proxies = {"http": None, "https": None}
    session.headers.update(_EM_HEADERS)
    for url in [
        "http://push2.eastmoney.com/api/qt/stock/get",
        "https://push2.eastmoney.com/api/qt/stock/get",
    ]:
        try:
            r = session.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if not data or data.get("rc") != 0:
                continue
            raw = data.get("data") or {}
            info = {cn: raw[k] for k, cn in _CODE_MAP.items() if k in raw and raw[k] is not None}
            if info:
                logger.info("[Tool:fundamental] 数据源1成功: 获取 {} 个字段", len(info))
                return info
        except Exception:
            continue
    raise RuntimeError("数据源1(东方财富直连)未返回有效数据")


def _fetch_source2_efinance(symbol: str) -> dict:
    """数据源 2：efinance 单股基本信息"""
    logger.info("[Tool:fundamental] 尝试数据源2(efinance): {}", symbol)
    import efinance as ef
    s = ef.stock.get_base_info(symbol)
    if s is None or len(s) == 0:
        raise RuntimeError("数据源2(efinance)返回空")
    code_val = s.get("股票代码")
    if code_val is None or (isinstance(code_val, float) and str(code_val) == "nan") or str(code_val).strip() == "":
        raise RuntimeError("数据源2(efinance)无有效股票代码")
    logger.info("[Tool:fundamental] 数据源2成功")
    return {"_series": s, "_source": "efinance"}


def _fetch_source3_akshare(symbol: str, timeout: float = 10) -> dict:
    """数据源 3：akshare 单股接口"""
    logger.info("[Tool:fundamental] 尝试数据源3(akshare): {}", symbol)
    info_df = ak.stock_individual_info_em(symbol=symbol, timeout=timeout)
    if info_df is None or info_df.empty or "item" not in info_df.columns or "value" not in info_df.columns:
        raise RuntimeError("数据源3(akshare单股)返回空")
    info = dict(zip(info_df["item"], info_df["value"]))
    if not info:
        raise RuntimeError("数据源3(akshare单股)无有效字段")
    logger.info("[Tool:fundamental] 数据源3成功: 获取 {} 个字段", len(info))
    return {"_dict": info, "_source": "akshare"}


def _build_fundamental_err_msg(e: Exception) -> str:
    msg = str(e)
    if "RemoteDisconnected" in msg or "Connection aborted" in msg:
        return "连接被服务器断开，可能为网络限流或需验证。请稍后重试。"
    return f"获取基本面数据失败：{msg}。请确认股票代码正确且网络正常。"


def _lines_from_direct_em(info: dict, stock_code: str, _fmt_num, _fmt_int, _fmt_date) -> list:
    name = info.get("股票简称", "") or info.get("股票代码", stock_code)
    latest = info.get("最新", "")
    total_mv = info.get("总市值", "")
    return [
        f"【{name}（{stock_code}）基本面与行情】", "",
        "一、行情与估值",
        f"  最新价：{latest} 元",
        f"  总市值：{_fmt_num(total_mv) or total_mv}",
        f"  流通市值：{_fmt_num(info.get('流通市值')) or info.get('流通市值', '')}",
        f"  总股本：{_fmt_int(info.get('总股本')) or info.get('总股本', '')}",
        f"  流通股：{_fmt_int(info.get('流通股')) or info.get('流通股', '')}",
        f"  行业：{info.get('行业', '')}",
        f"  上市时间：{_fmt_date(info.get('上市时间'))}",
        "", "二、财务指标（最近报告期）",
    ]


def _lines_from_efinance(payload: dict, stock_code: str, _fmt_num, _fmt_int, _fmt_date) -> list:
    s = payload["_series"]

    def _sv(val, default="—"):
        if val is None or (isinstance(val, float) and str(val) == "nan"):
            return default
        return val if isinstance(val, str) else str(val)

    name = _sv(s.get("股票名称")) or _sv(s.get("股票代码")) or stock_code
    total_mv = s.get("总市值")
    return [
        f"【{name}（{stock_code}）基本面与行情】", "",
        "一、行情与估值",
        "  最新价：—（本接口未返回）",
        f"  总市值：{_fmt_num(total_mv) or _sv(total_mv) or '—'}",
        f"  流通市值：{_fmt_num(s.get('流通市值')) or _sv(s.get('流通市值')) or '—'}",
        f"  市盈率(动)：{_sv(s.get('市盈率(动)'))}",
        f"  市净率：{_sv(s.get('市净率'))}",
        f"  所处行业：{_sv(s.get('所处行业'))}",
        f"  ROE：{_sv(s.get('ROE'))}",
        f"  净利率：{_sv(s.get('净利率'))}",
        f"  毛利率：{_sv(s.get('毛利率'))}",
        "", "二、财务指标（最近报告期）",
    ]


def _lines_from_akshare(payload: dict, stock_code: str, _fmt_num, _fmt_int, _fmt_date) -> list:
    info = payload["_dict"]
    name = info.get("股票简称", "") or info.get("股票代码", stock_code)
    latest = info.get("最新", "")
    total_mv = info.get("总市值", "")
    return [
        f"【{name}（{stock_code}）基本面与行情】", "",
        "一、行情与估值",
        f"  最新价：{latest} 元",
        f"  总市值：{_fmt_num(total_mv) or total_mv}",
        f"  流通市值：{_fmt_num(info.get('流通市值')) or info.get('流通市值', '')}",
        f"  总股本：{_fmt_int(info.get('总股本')) or info.get('总股本', '')}",
        f"  流通股：{_fmt_int(info.get('流通股')) or info.get('流通股', '')}",
        f"  行业：{info.get('行业', '')}",
        f"  上市时间：{_fmt_date(info.get('上市时间'))}",
        "", "二、财务指标（最近报告期）",
    ]


@tool
def get_fundamental_data(stock_code: str) -> str:
    """
    获取 A 股股票的基本面数据，包括市盈率、市净率、最新价、涨跌幅、换手率及财务指标（净利率、ROE 等）。
    当用户询问某只股票的基本面、财务、估值时调用此工具。
    采用多数据源 + 限频 + 降级：三路数据源各尝试一次。

    Args:
        stock_code: 6 位股票代码，例如 "300567"（精测电子）、"600519"（贵州茅台）。

    Returns:
        格式化的基本面与行情文本；若未找到或出错则返回说明文字。
    """
    logger.info("[Tool:fundamental] ▶ 被调用: stock_code={}", stock_code)
    start = time.time()
    _rate_limit()

    code_str = str(stock_code).strip()
    if len(code_str) == 5:
        code_str = "0" + code_str

    def _fmt_num(x):
        try:
            return f"{float(x) / 1e8:.2f}亿" if x is not None and str(x).strip() else ""
        except (TypeError, ValueError):
            return ""

    def _fmt_int(x):
        try:
            return f"{int(float(x)):,}" if x is not None and str(x).strip() else ""
        except (TypeError, ValueError):
            return ""

    def _fmt_date(x):
        s = str(x).strip() if x else ""
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) >= 8 else s

    lines = []
    last_error = None

    try:
        info = _fetch_source1_direct_em(code_str)
        lines = _lines_from_direct_em(info, stock_code, _fmt_num, _fmt_int, _fmt_date)
    except Exception as e1:
        logger.warning("[Tool:fundamental] 数据源1失败: {}", e1)
        last_error = e1
        time.sleep(_FALLBACK_DELAY_SEC)
        try:
            payload = _fetch_source2_efinance(code_str)
            lines = _lines_from_efinance(payload, stock_code, _fmt_num, _fmt_int, _fmt_date)
        except Exception as e2:
            logger.warning("[Tool:fundamental] 数据源2失败: {}", e2)
            last_error = e2
            time.sleep(_FALLBACK_DELAY_SEC)
            try:
                payload = _fetch_source3_akshare(code_str)
                lines = _lines_from_akshare(payload, stock_code, _fmt_num, _fmt_int, _fmt_date)
            except Exception as e3:
                logger.error("[Tool:fundamental] 全部数据源失败: {}", e3)
                last_error = e3
                return _build_fundamental_err_msg(last_error)

    symbol_em = f"{code_str}.SH" if code_str.startswith("6") else f"{code_str}.SZ"
    try:
        logger.info("[Tool:fundamental] 获取财务指标: {}", symbol_em)
        fin_df = ak.stock_financial_analysis_indicator_em(symbol=symbol_em, indicator="按报告期")
        if fin_df is not None and not fin_df.empty:
            latest_report = fin_df.iloc[0]
            skip_cols = {"SECUCODE", "SECURITY_CODE", "ORG_CODE"}
            shown = 0
            for col in fin_df.columns:
                if col in skip_cols or shown >= 10:
                    continue
                val = latest_report.get(col)
                if val is not None and str(val).strip() not in ("", "nan", "None"):
                    lines.append(f"  {col}：{val}")
                    shown += 1
            logger.info("[Tool:fundamental] 财务指标获取成功: {} 个字段", shown)
        else:
            lines.append("  （暂无或接口未返回数据）")
    except Exception as fin_err:
        logger.warning("[Tool:fundamental] 财务指标获取失败: {}", fin_err)
        lines.append("  （暂无或接口未返回数据）")

    elapsed = time.time() - start
    result = "\n".join(lines).strip()
    logger.info("[Tool:fundamental] ✅ 完成: stock={}, 输出{}字, 耗时={:.1f}s",
                stock_code, len(result), elapsed)
    return result
