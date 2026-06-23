from contextvars import ContextVar
from copy import deepcopy
from typing import Any, Dict, List, Optional


_ACTIVE_SESSION_ID: ContextVar[str] = ContextVar("active_session_id", default="default")
_SESSION_STORE: Dict[str, Dict[str, Any]] = {}

_MAX_TOOL_TRACE = 50
_MAX_CONVERSATION_HISTORY = 20
_MAX_TOPIC_MEMORY = 12
_MAX_RECENT_ORDER_IDS = 10


def _normalize_order_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.upper()


def _dedupe_preserve_order(values: List[Any]) -> List[Any]:
    seen = set()
    result = []
    for value in values:
        marker = repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result


def _normalize_topic_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {
        "domain": str(entry.get("domain") or "").strip(),
        "label": str(entry.get("label") or "").strip(),
        "data": deepcopy(entry.get("data") or {}),
    }

    if "order_id" in normalized["data"]:
        normalized["data"]["order_id"] = _normalize_order_id(normalized["data"]["order_id"])

    if "query" in normalized["data"] and normalized["data"]["query"] is not None:
        normalized["data"]["query"] = str(normalized["data"]["query"]).strip()

    if "product_query" in normalized["data"] and normalized["data"]["product_query"] is not None:
        normalized["data"]["product_query"] = str(normalized["data"]["product_query"]).strip()

    if "brands" in normalized["data"] and isinstance(normalized["data"]["brands"], list):
        normalized["data"]["brands"] = _dedupe_preserve_order(
            [str(item).strip().lower() for item in normalized["data"]["brands"] if str(item).strip()]
        )

    if "categories" in normalized["data"] and isinstance(normalized["data"]["categories"], list):
        normalized["data"]["categories"] = _dedupe_preserve_order(
            [str(item).strip().lower() for item in normalized["data"]["categories"] if str(item).strip()]
        )

    return normalized


def _new_session_state() -> Dict[str, Any]:
    return {
        "tool_trace": [],
        "session_customer": None,
        "conversation_state": {
            "pending_intent": None,
            "last_order_id": None,
            "pending_offer": None,
            "last_product_query": None,
            "recent_order_ids": [],
            "topic_memory": [],
        },
        "conversation_history": [],
        "runtime_state": {
            "last_response_source": None,
            "last_llm_error": None,
        },
    }


def _get_state(session_id: Optional[str] = None) -> Dict[str, Any]:
    resolved_session_id = str(session_id or _ACTIVE_SESSION_ID.get() or "default")
    if resolved_session_id not in _SESSION_STORE:
        _SESSION_STORE[resolved_session_id] = _new_session_state()
    return _SESSION_STORE[resolved_session_id]


def set_active_session(session_id: str):
    resolved_session_id = str(session_id or "default")
    _ACTIVE_SESSION_ID.set(resolved_session_id)
    _get_state(resolved_session_id)


def get_active_session() -> str:
    return _ACTIVE_SESSION_ID.get()


def add_tool_trace(tool_name: str, input_data: Dict[str, Any], output_data: Any):
    state = _get_state()
    trace_entry = {
        "tool_name": str(tool_name),
        "input": deepcopy(input_data or {}),
        "output": deepcopy(output_data),
    }
    state["tool_trace"].append(trace_entry)

    if len(state["tool_trace"]) > _MAX_TOOL_TRACE:
        state["tool_trace"] = state["tool_trace"][-_MAX_TOOL_TRACE:]


def get_tool_trace() -> List[Dict[str, Any]]:
    return deepcopy(_get_state()["tool_trace"])


def get_tool_trace_length() -> int:
    return len(_get_state()["tool_trace"])


def get_tool_trace_since(index: int) -> List[Dict[str, Any]]:
    traces = _get_state()["tool_trace"]
    if index < 0:
        index = 0
    return deepcopy(traces[index:])


def clear_tool_trace():
    _get_state()["tool_trace"] = []


def set_session_customer(customer_id: int, display_name: str):
    _get_state()["session_customer"] = {
        "customer_id": int(customer_id),
        "display_name": str(display_name).strip(),
    }


def clear_session_customer():
    _get_state()["session_customer"] = None


def get_session_customer() -> Optional[Dict[str, Any]]:
    customer = _get_state()["session_customer"]
    return None if customer is None else deepcopy(customer)


def set_pending_intent(intent: Optional[str]):
    _get_state()["conversation_state"]["pending_intent"] = None if intent is None else str(intent).strip()


def get_pending_intent() -> Optional[str]:
    return _get_state()["conversation_state"]["pending_intent"]


def set_last_order_id(order_id: Optional[Any]):
    normalized_order_id = _normalize_order_id(order_id)
    _get_state()["conversation_state"]["last_order_id"] = normalized_order_id

    if normalized_order_id is not None:
        current_recent = get_recent_order_ids()
        updated = [normalized_order_id] + [item for item in current_recent if item != normalized_order_id]
        set_recent_order_ids(updated[:_MAX_RECENT_ORDER_IDS])


def get_last_order_id() -> Optional[str]:
    value = _get_state()["conversation_state"]["last_order_id"]
    return _normalize_order_id(value)


def clear_last_order_id():
    _get_state()["conversation_state"]["last_order_id"] = None


def set_pending_offer(offer: Optional[str]):
    _get_state()["conversation_state"]["pending_offer"] = None if offer is None else str(offer).strip()


def get_pending_offer() -> Optional[str]:
    return _get_state()["conversation_state"]["pending_offer"]


def set_last_product_query(query: Optional[str]):
    cleaned = None if query is None else str(query).strip()
    _get_state()["conversation_state"]["last_product_query"] = cleaned or None


def get_last_product_query() -> Optional[str]:
    return _get_state()["conversation_state"]["last_product_query"]


def clear_last_product_query():
    _get_state()["conversation_state"]["last_product_query"] = None


def set_recent_order_ids(order_ids: List[Any]):
    normalized = []
    for order_id in order_ids or []:
        clean = _normalize_order_id(order_id)
        if clean is not None:
            normalized.append(clean)

    normalized = _dedupe_preserve_order(normalized)[:_MAX_RECENT_ORDER_IDS]
    _get_state()["conversation_state"]["recent_order_ids"] = normalized


def get_recent_order_ids() -> List[str]:
    return list(_get_state()["conversation_state"]["recent_order_ids"])


def remember_topic(domain: str, label: str, data: Optional[Dict[str, Any]] = None):
    memory = _get_state()["conversation_state"]["topic_memory"]

    entry = _normalize_topic_entry({
        "domain": domain,
        "label": label,
        "data": data or {},
    })

    if not entry["domain"] or not entry["label"]:
        return

    filtered = [
        item for item in memory
        if not (
            item.get("domain") == entry["domain"] and
            item.get("label") == entry["label"] and
            item.get("data") == entry["data"]
        )
    ]

    filtered.append(entry)

    if len(filtered) > _MAX_TOPIC_MEMORY:
        filtered = filtered[-_MAX_TOPIC_MEMORY:]

    _get_state()["conversation_state"]["topic_memory"] = filtered


def get_topic_memory() -> List[Dict[str, Any]]:
    return deepcopy(_get_state()["conversation_state"]["topic_memory"])


def clear_topic_memory(domain: Optional[str] = None):
    if domain is None:
        _get_state()["conversation_state"]["topic_memory"] = []
        return

    domain = str(domain).strip()
    memory = _get_state()["conversation_state"]["topic_memory"]
    _get_state()["conversation_state"]["topic_memory"] = [
        item for item in memory if item.get("domain") != domain
    ]


def get_recent_topic(domain: Optional[str] = None) -> Optional[Dict[str, Any]]:
    memory = _get_state()["conversation_state"]["topic_memory"]
    for item in reversed(memory):
        if domain is None or item.get("domain") == domain:
            return deepcopy(item)
    return None


def get_recent_tool_trace(limit: int = 4) -> List[Dict[str, Any]]:
    traces = _get_state()["tool_trace"]
    if limit <= 0:
        return []
    return deepcopy(traces[-limit:])


def add_conversation_turn(role: str, content: str):
    role = str(role).strip()
    content = str(content)

    history = _get_state()["conversation_history"]

    if history and history[-1]["role"] == role and history[-1]["content"] == content:
        return

    history.append({
        "role": role,
        "content": content,
    })

    if len(history) > _MAX_CONVERSATION_HISTORY:
        _get_state()["conversation_history"] = history[-_MAX_CONVERSATION_HISTORY:]


def get_conversation_history() -> List[Dict[str, str]]:
    return deepcopy(_get_state()["conversation_history"])


def clear_conversation_history():
    _get_state()["conversation_history"] = []


def clear_pending_state():
    conversation_state = _get_state()["conversation_state"]
    conversation_state["pending_intent"] = None
    conversation_state["pending_offer"] = None


def reset_conversation_state():
    _get_state()["conversation_state"] = {
        "pending_intent": None,
        "last_order_id": None,
        "pending_offer": None,
        "last_product_query": None,
        "recent_order_ids": [],
        "topic_memory": [],
    }


def get_session_snapshot() -> Dict[str, Any]:
    state = _get_state()
    return {
        "session_id": get_active_session(),
        "session_customer": deepcopy(state["session_customer"]),
        "conversation_state": deepcopy(state["conversation_state"]),
        "tool_trace_length": len(state["tool_trace"]),
        "conversation_history": deepcopy(state["conversation_history"]),
        "runtime_state": deepcopy(state["runtime_state"]),
    }


def reset_session(session_id: Optional[str] = None):
    resolved_session_id = str(session_id or get_active_session() or "default")
    _SESSION_STORE[resolved_session_id] = _new_session_state()


def set_last_response_source(source: Optional[str]):
    _get_state()["runtime_state"]["last_response_source"] = None if source is None else str(source).strip()


def get_last_response_source() -> Optional[str]:
    return _get_state()["runtime_state"]["last_response_source"]


def set_last_llm_error(error_message: Optional[str]):
    _get_state()["runtime_state"]["last_llm_error"] = None if error_message is None else str(error_message).strip()


def get_last_llm_error() -> Optional[str]:
    return _get_state()["runtime_state"]["last_llm_error"]