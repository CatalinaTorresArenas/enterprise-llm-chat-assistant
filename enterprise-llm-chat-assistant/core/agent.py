import os
import re
import sys
import json
import random
import unicodedata
from datetime import datetime
from typing import Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.router import classify_intent as classify_intent_rules
from core.guards import (
    require_auth_if_needed,
    detect_prompt_injection,
    validate_tool_usage,
)
from core.session_context import (
    add_conversation_turn,
    get_conversation_history,
    get_last_llm_error,
    get_last_response_source,
    get_recent_tool_trace,
    get_session_snapshot,
    set_last_llm_error,
    set_last_response_source,
    set_active_session,
    reset_session,
    get_tool_trace_length,
    get_tool_trace_since,
    get_session_customer,
    get_pending_intent,
    set_pending_intent,
    get_last_order_id,
    set_last_order_id,
    get_pending_offer,
    set_pending_offer,
    get_last_product_query,
    set_last_product_query,
    set_recent_order_ids,
    remember_topic,
    get_recent_topic,
    get_topic_memory,
)
from core.llm_client import build_llm_client, get_llm_runtime_metadata, LLMClientError
from core.response_validator import validate_hybrid_response
from core.tool_registry import execute_tool, get_tool_definitions_for_llm

_AGENT_BOOT_ERROR = None

try:
    from core.tools import (
        extract_customer_identifier,
        authenticate_customer,
        get_order_status,
        get_order_amounts,
        get_order_items,
        get_order_payment_method,
        get_order_history,
        get_shipment_details,
        get_product_warranty_info,
        search_products,
        search_promotions,
        list_product_categories,
        get_catalog_reference_terms,
        get_customer_orders_summary,
        get_customer_orders_for_selection,
        get_customer_default_address,
        get_order_delivery_address,
    )
except Exception as exc:
    _AGENT_BOOT_ERROR = exc

    def _tool_import_error(*args, **kwargs):
        raise RuntimeError("Tool layer unavailable") from exc

    extract_customer_identifier = _tool_import_error
    authenticate_customer = _tool_import_error
    get_order_status = _tool_import_error
    get_order_amounts = _tool_import_error
    get_order_items = _tool_import_error
    get_order_payment_method = _tool_import_error
    get_order_history = _tool_import_error
    get_shipment_details = _tool_import_error
    get_product_warranty_info = _tool_import_error
    search_products = _tool_import_error
    search_promotions = _tool_import_error
    list_product_categories = _tool_import_error
    get_catalog_reference_terms = _tool_import_error
    get_customer_orders_summary = _tool_import_error
    get_customer_orders_for_selection = _tool_import_error
    get_customer_default_address = _tool_import_error
    get_order_delivery_address = _tool_import_error

try:
    from core.policy_rag import search_policy_sections, format_policy_response
except Exception as exc:
    if _AGENT_BOOT_ERROR is None:
        _AGENT_BOOT_ERROR = exc

    def _policy_import_error(*args, **kwargs):
        raise RuntimeError("Policy layer unavailable") from exc

    search_policy_sections = _policy_import_error
    format_policy_response = _policy_import_error


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", (text or "").lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _extract_order_id(message: str) -> Optional[str]:
    raw = (message or "").strip()

    contextual_patterns = [
        r"(?:pedido|orden|order)\s*(?:numero|nro|no)?\s*#?\s*([A-Za-z]{0,4}-?\d{1,})",
        r"#\s*([A-Za-z]{0,4}-?\d{1,})",
        r"\b([A-Za-z]{1,4}-?\d{3,})\b",
        r"\b(\d{3,})\b",
    ]
    for pattern in contextual_patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            return match.group(1).replace(" ", "").upper()

    return None


def _extract_identifier(message: str) -> Optional[str]:
    return extract_customer_identifier(message)


def _classify_intent(
    message: str,
    session_customer=None,
    pending_intent: Optional[str] = None,
    pending_offer: Optional[str] = None,
):
    return classify_intent_rules(message)


def _finalize_response(user_message: str, intent: str, response: str) -> str:
    return response.strip()


def _extract_product_focus_query(message: str) -> str:
    normalized = _normalize_text(message)
    catalog_terms = sorted(get_catalog_reference_terms(), key=len, reverse=True)
    matches = []
    for term in catalog_terms:
        if term in normalized and term not in matches:
            matches.append(term)

    if matches:
        return " ".join(matches[:3])

    return normalized


def _references_previous_product_context(message: str) -> bool:
    normalized = _normalize_text(message)
    patterns = [
        "ese", "esa", "esos", "esas", "este", "estos",
        "para los", "para esas", "para esos",
        "de esos", "de esas", "de ese", "sobre eso",
    ]
    return any(pattern in normalized for pattern in patterns)


def _extract_brand_mentions(message: str) -> list[str]:
    normalized = _normalize_text(message)
    known_brands = ["samsung", "apple", "iphone", "xiaomi", "sony", "lg", "lenovo", "hp", "asus", "huawei"]
    return [brand for brand in known_brands if re.search(rf"\b{re.escape(brand)}\b", normalized)]


def _extract_contextual_brand_mentions(message: str) -> list[str]:
    normalized = _normalize_text(message)
    mentions = list(_extract_brand_mentions(message))
    recent_product_topic = get_recent_topic("product")
    if recent_product_topic:
        for brand in recent_product_topic.get("data", {}).get("brands", []):
            normalized_brand = _normalize_text(brand)
            if normalized_brand and re.search(rf"\b{re.escape(normalized_brand)}\b", normalized):
                mentions.append(normalized_brand)

    deduped = []
    for mention in mentions:
        if mention not in deduped:
            deduped.append(mention)
    return deduped


def _is_context_recall_message(message: str) -> bool:
    normalized = _normalize_text(message)
    patterns = [
        "recuerda que te hable",
        "recuerda que hablamos",
        "te hable de",
        "te hablé de",
        "te dije que buscaba",
        "te comente que buscaba",
        "te comenté que buscaba",
        "volviendo a eso",
        "retomando eso",
        "de eso",
        "de lo anterior",
        "de ese tema",
        "de ese producto",
        "de esa promocion",
        "de ese pedido",
        "de lo que hablamos",
    ]
    return any(pattern in normalized for pattern in patterns)


def _build_contextual_product_query(message: str, last_product_query: Optional[str]) -> str:
    normalized = _normalize_text(message)
    brand_mentions = _extract_contextual_brand_mentions(message)
    recommendation_followup = _is_product_recommendation_query(message)

    recent_product_topic = get_recent_topic("product")
    active_categories = []
    if recent_product_topic:
        active_categories = recent_product_topic.get("data", {}).get("categories", []) or []

    explicit_catalog_terms = [term for term in get_catalog_reference_terms() if term in normalized]
    if explicit_catalog_terms:
        return message

    base_parts = []
    if recommendation_followup and last_product_query:
        base_parts.append(last_product_query)
    elif active_categories:
        base_parts.extend(active_categories[:2])
    elif last_product_query:
        base_parts.append(last_product_query)

    if brand_mentions:
        base_parts.extend(brand_mentions)

    if _references_previous_product_context(message):
        base_parts.append(message)

    query = " ".join(part for part in base_parts if part).strip()
    return query or message


def _build_topic_recall_response() -> Optional[str]:
    topic_memory = get_topic_memory()
    if not topic_memory:
        return None

    recent = topic_memory[-3:]
    labels = [item.get("label", "ese tema") for item in recent]
    labels = list(dict.fromkeys(label for label in labels if label))
    if not labels:
        return None

    if len(labels) == 1:
        summary = labels[0]
    elif len(labels) == 2:
        summary = f"{labels[0]} y {labels[1]}"
    else:
        summary = f"{labels[0]}, {labels[1]} y {labels[2]}"

    return (
        f"Tengo presente que veníamos hablando de {summary}. "
        "Si quieres, retomamos cualquiera de esos temas y sigo desde ahí."
    )


def _build_contextual_promotion_query(message: str, last_product_query: Optional[str]) -> str:
    if not last_product_query:
        return message

    if _is_general_promotion_query(message):
        return f"{last_product_query} promociones"

    if _references_previous_product_context(message):
        return f"{last_product_query} {message}"

    brand_mentions = _extract_contextual_brand_mentions(message)
    if brand_mentions:
        return f"{last_product_query} {' '.join(brand_mentions)} promociones"

    return message


def _display_product_context(last_product_query: Optional[str]) -> str:
    normalized = _normalize_text(last_product_query or "")
    if not normalized:
        return "eso"

    tokens = []
    seen_roots = set()
    for token in normalized.split():
        if token.endswith("es") and len(token) > 4:
            root = token[:-2]
        elif token.endswith("s") and len(token) > 4:
            root = token[:-1]
        else:
            root = token
        if root in seen_roots:
            continue
        seen_roots.add(root)
        tokens.append(token)

    return " ".join(tokens[:2]) if tokens else "eso"


def _filter_promotion_result_by_context(
    result: dict,
    context_query: Optional[str],
    last_product_query: Optional[str] = None,
) -> dict:
    if not result.get("success") or not context_query:
        return result

    brand_terms = _extract_contextual_brand_mentions(context_query)

    recent_product_topic = get_recent_topic("product")
    category_terms = []
    if recent_product_topic:
        category_terms.extend(recent_product_topic.get("data", {}).get("categories", []))
    if not category_terms:
        category_terms.extend(_infer_categories_from_brand_mentions(context_query))
    if last_product_query:
        normalized_last_query = _normalize_text(last_product_query)
        category_terms.extend(
            term for term in get_catalog_reference_terms()
            if term in normalized_last_query
        )
    category_terms = list(dict.fromkeys(term for term in category_terms if term))

    if not brand_terms and not category_terms:
        return result

    filtered_items = []
    category_only_items = []
    for item in result.get("results", []):
        raw_targets = item.get("targets") or []
        targets = []
        for target in raw_targets:
            if isinstance(target, list):
                targets.extend(str(value) for value in target)
            else:
                targets.append(str(target))
        blob = _normalize_text(
            f"{item.get('promotion_name', '')} {item.get('description', '')} {' '.join(targets)}"
        )
        has_brand_match = not brand_terms or any(term in blob for term in brand_terms)
        has_category_match = not category_terms or any(term in blob for term in category_terms)
        if has_brand_match and has_category_match:
            filtered_items.append(item)
        elif has_category_match:
            category_only_items.append(item)

    if filtered_items:
        filtered_result = dict(result)
        filtered_result["results"] = filtered_items
        return filtered_result

    if category_only_items:
        filtered_result = dict(result)
        filtered_result["results"] = category_only_items
        return filtered_result

    return result


def _is_card_secret_request(message: str) -> bool:
    normalized = _normalize_text(message)
    patterns = [
        "numero completo de la tarjeta",
        "numero de la tarjeta completo",
        "numero completo tarjeta",
        "dime la tarjeta completa",
        "tarjeta completa",
        "cvv",
        "cvc",
        "codigo de seguridad",
        "codigo de verificacion",
        "fecha de vencimiento",
        "expiracion de la tarjeta",
        "expiracion",
        "datos completos de la tarjeta",
    ]
    return any(pattern in normalized for pattern in patterns)


def _clear_pending_flow():
    set_pending_intent(None)
    set_pending_offer(None)
    set_last_product_query(None)


def _clear_auth_pending_state():
    set_pending_intent(None)
    set_pending_offer(None)


def _get_memory_product_query() -> Optional[str]:
    current_query = get_last_product_query()
    if current_query:
        return current_query

    recent_product_topic = get_recent_topic("product")
    if recent_product_topic:
        return recent_product_topic.get("data", {}).get("query")

    recent_promotion_topic = get_recent_topic("promotion")
    if recent_promotion_topic:
        return recent_promotion_topic.get("data", {}).get("product_query")

    return None


def _get_recent_product_query_for_promotion_context() -> Optional[str]:
    topic_memory = get_topic_memory()
    for item in reversed(topic_memory):
        domain = item.get("domain")
        data = item.get("data", {})
        if domain == "product":
            query = data.get("query")
            if query:
                return str(query)
        if domain == "promotion":
            product_query = data.get("product_query")
            if product_query:
                return str(product_query)
    return _get_memory_product_query()


def _get_memory_order_id() -> Optional[str]:
    current_order_id = get_last_order_id()
    if current_order_id is not None:
        return current_order_id

    recent_order_topic = get_recent_topic("order")
    if recent_order_topic:
        return recent_order_topic.get("data", {}).get("order_id")

    return None


def _remember_product_topic(
    query: Optional[str],
    label: Optional[str] = None,
    brands: Optional[list[str]] = None,
    categories: Optional[list[str]] = None,
):
    if not query:
        return
    topic_label = label or _display_product_context(query)
    remember_topic(
        "product",
        topic_label,
        {
            "query": query,
            "brands": list(brands or []),
            "categories": list(categories or []),
        },
    )


def _remember_promotion_topic(query: str, product_query: Optional[str] = None):
    remember_topic(
        "promotion",
        _display_product_context(product_query or query),
        {
            "query": query,
            "product_query": product_query,
        },
    )


def _remember_order_topic(order_id: Optional[str], intent: Optional[str] = None):
    if order_id is None:
        return
    remember_topic(
        "order",
        f"pedido {order_id}",
        {
            "order_id": order_id,
            "intent": intent,
        },
    )


def _remember_policy_topic(query: str):
    remember_topic("policy", "politicas", {"query": query})


def _extract_topic_brands_and_categories(result: dict) -> tuple[list[str], list[str]]:
    if not result.get("success"):
        return [], []

    brands = []
    categories = []
    for item in result.get("results", []):
        brand = item.get("brand")
        category = item.get("category")
        if brand:
            normalized_brand = _normalize_text(str(brand))
            if normalized_brand and normalized_brand not in brands:
                brands.append(normalized_brand)
        if category:
            normalized_category = _normalize_text(str(category))
            if normalized_category and normalized_category not in categories:
                categories.append(normalized_category)
    return brands, categories


def _infer_categories_from_brand_mentions(message: str) -> list[str]:
    brand_mentions = _extract_contextual_brand_mentions(message)
    candidate_queries = list(brand_mentions)
    if not candidate_queries:
        normalized = _normalize_text(message)
        removable_terms = {
            "los", "las", "el", "la", "un", "una", "unos", "unas", "tienen", "tiene",
            "algun", "alguna", "descuento", "descuentos", "promocion", "promociones",
            "oferta", "ofertas", "rebaja", "rebajas", "quiero", "quisiera", "estoy",
            "interesada", "interesado", "en", "de", "para", "alguna",
        }
        tokens = [token for token in normalized.split() if token not in removable_terms]
        if tokens:
            candidate_queries.append(" ".join(tokens[:3]))
    if not candidate_queries:
        return []

    inferred_categories = []
    for candidate in candidate_queries:
        product_result = search_products(candidate)
        _, categories = _extract_topic_brands_and_categories(product_result)
        for category in categories:
            if category not in inferred_categories:
                inferred_categories.append(category)
    return inferred_categories


def _validate_or_return(intent: str, trace_start_index: int, response: str) -> str:
    tool_validation = validate_tool_usage(
        {"intent": intent, "requires_tool": True},
        trace_start_index
    )
    if not tool_validation["valid"]:
        return tool_validation["message"]
    return response


def _greeting_response() -> str:
    return (
        "Hola, ¿en qué puedo ayudarte?\n"
        "Puedo ayudarte con pedidos, productos y políticas de la tienda. "
        "Si quieres revisar un pedido, puedes compartirme tu cédula o número de teléfono."
    )


def _faq_response(message: str) -> str:
    msg = _normalize_text(message)

    if _is_purchase_guidance_query(message):
        return (
            "Si quieres hacer una compra, primero dime qué producto buscas y yo te muestro opciones "
            "con precio, stock y tiempos de entrega. Si ya tienes una referencia en mente, también "
            "puedo ayudarte a revisarla y orientarte con la compra."
        )

    if "pago" in msg:
        return (
            "Manejamos métodos de pago como tarjeta crédito o débito, PSE, "
            "contraentrega, Nequi y Daviplata."
        )

    if "cobertura" in msg or "envio" in msg:
        return (
            "Tenemos cobertura de envíos a todo el territorio colombiano. "
            "Si quieres, también puedo ayudarte con tiempos o condiciones de entrega."
        )

    if "canales" in msg or "atencion" in msg or "contacto" in msg:
        return (
            "Puedo ayudarte con información general de atención. "
            "Si me dices qué necesitas, te respondo de forma más específica."
        )

    return "Puedo ayudarte con preguntas generales, políticas, productos o pedidos."


def _is_purchase_guidance_query(message: str) -> bool:
    normalized = _normalize_text(message)
    patterns = [
        "como puedo comprar",
        "como comprar",
        "como puedo hacer la compra",
        "como hago la compra",
        "hacer la compra",
        "quiero comprar",
        "como hago para comprar",
        "como hago para pedir",
        "como hago un pedido",
        "como puedo hacer un pedido",
    ]
    return any(pattern in normalized for pattern in patterns)


def _is_product_recommendation_query(message: str) -> bool:
    normalized = _normalize_text(message)
    recommendation_terms = [
        "me recomiendas",
        "me recomienda",
        "cual me recomiendas",
        "cual me recomienda",
        "que me recomiendas",
        "que me recomienda",
        "cual nevera me recomiendas",
        "cual producto me recomiendas",
        "cual me conviene",
        "que me conviene",
        "te parece mejor",
    ]
    budget_terms = [
        "presupuesto",
        "tengo ",
        "hasta ",
        "maximo",
        "maximo de",
        "millon",
        "millones",
        "$",
    ]
    has_recommendation_term = any(term in normalized for term in recommendation_terms)
    has_budget_term = any(term in normalized for term in budget_terms)
    return has_recommendation_term or (has_budget_term and any(word in normalized for word in ["recomi", "conviene"]))


def _extract_budget_amount(message: str) -> Optional[float]:
    raw_message = (message or "").strip()
    normalized = _normalize_text(raw_message)

    currency_pattern = re.search(r"\$\s*([\d\.\,]+)", raw_message)
    if currency_pattern:
        raw_amount = currency_pattern.group(1).replace(".", "").replace(",", "")
        if raw_amount.isdigit():
            return float(raw_amount)

    million_pattern = re.search(r"(\d+(?:[\.,]\d+)?)\s*millones?", normalized)
    if million_pattern:
        return float(million_pattern.group(1).replace(",", ".")) * 1_000_000

    mil_pattern = re.search(r"(\d+(?:[\.,]\d+)?)\s*mil\b", normalized)
    if mil_pattern:
        return float(mil_pattern.group(1).replace(",", ".")) * 1_000

    plain_number_pattern = re.search(r"\b(\d{5,})\b", normalized)
    if plain_number_pattern:
        return float(plain_number_pattern.group(1))

    return None


def _format_money(value: float) -> str:
    return f"${value:,.0f}".replace(",", ".")


def _score_recommended_product(item: dict, budget: Optional[float]) -> tuple:
    within_budget = budget is None or item["price"] <= budget
    budget_gap = abs((budget or item["price"]) - item["price"])
    delivery_days = item.get("shipping_days") or 999

    # Priorizamos referencias activas y con mejor experiencia de compra.
    return (
        1 if within_budget else 0,
        1 if item.get("active") else 0,
        0 if item.get("is_final_sale") else 1,
        1 if item.get("free_shipping") else 0,
        1 if item.get("available_qty", 0) > 0 else 0,
        -budget_gap,
        -item.get("available_qty", 0),
        -delivery_days,
    )


def _format_product_recommendation_response(message: str, result: dict) -> str:
    if not result["success"] or not result.get("results"):
        return result["message"]

    budget = _extract_budget_amount(message)
    items = result["results"]
    within_budget = [item for item in items if budget is None or item["price"] <= budget]
    candidate_pool = within_budget or items

    ranked_candidates = sorted(
        candidate_pool,
        key=lambda item: _score_recommended_product(item, budget),
        reverse=True,
    )
    best_item = ranked_candidates[0]

    reasons = []
    if budget is not None and best_item["price"] <= budget:
        reasons.append(f"entra en tu presupuesto de {_format_money(budget)}")
    if best_item.get("active"):
        reasons.append("es una referencia activa")
    if best_item.get("free_shipping"):
        reasons.append("tiene envío gratis")
    if best_item.get("available_qty", 0) > 0:
        reasons.append(f"hay {best_item['available_qty']} disponibles")
    if best_item.get("shipping_days"):
        reasons.append(f"la entrega estimada es de {best_item['shipping_days']} días")

    caution_notes = []
    if not best_item.get("active"):
        caution_notes.append("ojo: esa referencia aparece inactiva")
    if best_item.get("is_final_sale"):
        caution_notes.append("además aplica como venta final")
    if best_item.get("requires_installation"):
        caution_notes.append("ten presente que requiere instalación")

    alternative = next((item for item in ranked_candidates[1:] if item.get("active")), None)
    if alternative is None:
        alternative = next((item for item in items if item["product_id"] != best_item["product_id"]), None)

    intro = f"Te recomendaría la {best_item['name']}"
    if reasons:
        intro += " porque " + ", ".join(reasons)
    intro += "."

    lines = [intro, f"Queda en {_format_money(best_item['price'])}."]

    if caution_notes:
        lines.append(" ".join(caution_notes).capitalize() + ".")

    if alternative is not None:
        lines.append(
            f"Como segunda opción, miraría la {alternative['name']}, que queda en {_format_money(alternative['price'])}"
            + (f" y llega en {alternative['shipping_days']} días." if alternative.get("shipping_days") else ".")
        )

    lines.append("Si quieres, te la comparo frente a las otras por precio, entrega y si conviene evitar las inactivas.")
    return "\n".join(lines)


def _wants_last_order(message: str) -> bool:
    msg = _normalize_text(message)
    patterns = [
        "ultimo pedido",
        "pedido mas reciente",
        "mas reciente",
        "mi ultimo pedido",
    ]
    return any(p in msg for p in patterns)


def _wants_order_by_status(message: str) -> Optional[str]:
    msg = _normalize_text(message)

    if "cancelado" in msg or "cancelada" in msg:
        return "cancelled"
    if "entregado" in msg or "entregada" in msg:
        return "delivered"
    if "enviado" in msg or "despachado" in msg:
        return "shipped"
    if "devuelto" in msg or "devuelta" in msg:
        return "returned"

    return None


def _resolve_selected_order_intent(intent: str, order_id: str) -> str:
    set_last_order_id(order_id)
    _remember_order_topic(order_id, intent)

    if intent == "order_amount":
        result = get_order_amounts(order_id)
        _clear_pending_flow()
        return _format_order_amounts_response(result)

    if intent == "order_items":
        result = get_order_items(order_id)
        _clear_pending_flow()
        return _format_order_items_response(result)

    if intent == "order_payment_method":
        result = get_order_payment_method(order_id)
        _clear_pending_flow()
        return _format_payment_method_response(result)

    if intent == "order_delivery_address":
        result = get_order_delivery_address(order_id)
        _clear_pending_flow()
        return _format_order_delivery_address_response(result)

    status_result = get_order_status(order_id)
    history_result = get_order_history(order_id)
    shipment_result = get_shipment_details(order_id)

    if intent == "returns_order_case":
        _clear_pending_flow()
        return _format_returns_case_response(history_result, shipment_result, status_result)

    return _format_order_summary_options(status_result, shipment_result)


def _resolve_recent_order_after_auth(intent: str) -> str:
    selection_result = get_customer_orders_for_selection(limit=10)
    selected_order_id = _select_order_from_summary(selection_result, "ultimo pedido")

    if selected_order_id is None:
        if selection_result["success"] and selection_result["results"]:
            selected_order_id = str(selection_result["results"][0]["order_id"])
        else:
            _clear_pending_flow()
            return _format_customer_orders_summary(selection_result)

    if intent == "order_status_history":
        set_last_order_id(selected_order_id)
        status_result = get_order_status(selected_order_id)
        shipment_result = get_shipment_details(selected_order_id)
        _remember_order_topic(selected_order_id, "order_status_history")
        _clear_pending_flow()
        set_pending_offer("order_summary_options")
        return _format_order_summary_options(status_result, shipment_result)

    return _resolve_selected_order_intent(intent, selected_order_id)


def _normalize_recent_order_intent(intent: Optional[str]) -> str:
    supported_intents = {
        "order_amount",
        "order_items",
        "order_payment_method",
        "order_delivery_address",
        "order_status_history",
        "returns_order_case",
    }
    if intent in supported_intents:
        return intent
    return "order_status_history"


def _resolve_address_intent(message: str, last_order_id: Optional[str] = None) -> Optional[str]:
    msg = _normalize_text(message)

    delivery_patterns = [
        "direccion de entrega",
        "a que direccion",
        "para donde iba",
        "donde lo iban a entregar",
        "a donde lo iban a mandar",
        "a donde iba",
        "direccion del pedido",
        "direccion de ese pedido",
        "a donde iba el pedido",
    ]
    if any(pattern in msg for pattern in delivery_patterns):
        return "order_delivery_address" if last_order_id is not None else "customer_address"

    customer_patterns = [
        "cual es mi direccion",
        "mi direccion",
        "direccion registrada",
        "direccion principal",
        "direccion de mi cuenta",
    ]
    if any(pattern in msg for pattern in customer_patterns):
        return "customer_address"

    ambiguous_patterns = [
        "direccion",
        "la direccion",
        "y la direccion",
        "esa direccion",
        "cual es la direccion",
        "cual era la direccion",
        "donde era la direccion",
        "cual era esa direccion",
    ]
    if any(pattern in msg for pattern in ambiguous_patterns):
        return "order_delivery_address" if last_order_id is not None else "customer_address"

    return None


def _format_auth_failure_response() -> str:
    _clear_auth_pending_state()
    return (
        "No pude validar la cuenta con ese dato. "
        "Intenta nuevamente con una cédula o un teléfono que esté registrado."
    )


def _looks_like_customer_identifier_attempt(message: str) -> bool:
    normalized = _normalize_text(message)
    if re.fullmatch(r"[\+\-\s#a-z0-9]+", normalized) is None:
        return False

    has_digits = any(ch.isdigit() for ch in normalized)
    has_identifier_shape = bool(re.search(r"[a-z]{1,4}\s*-?\s*\d{3,}", normalized))
    compact = re.sub(r"[\s\-\+#]", "", normalized)
    return has_digits and (len(compact) >= 4 or has_identifier_shape)


def _is_auth_pending_state(pending_intent: Optional[str], pending_offer: Optional[str]) -> bool:
    auth_pending_offers = {
        "resolve_recent_order_after_auth",
        "show_recent_orders_after_auth",
    }
    auth_pending_intents = {
        "order_amount",
        "order_items",
        "order_payment_method",
        "order_delivery_address",
        "order_status_history",
        "returns_order_case",
        "customer_address",
    }
    return pending_offer in auth_pending_offers or pending_intent in auth_pending_intents


def _is_affirmative_reply(message: str) -> bool:
    normalized = _normalize_text(message)
    affirmatives = {
        "si", "sí", "dale", "ok", "bueno", "de una", "claro", "yes",
        "si dime", "sí dime", "dime", "cuentame", "cuéntame"
    }
    return normalized in affirmatives


def _is_product_filter_followup(message: str) -> bool:
    normalized = _normalize_text(message)
    patterns = [
        "por disponibilidad",
        "disponibilidad",
        "solo disponibles",
        "solo los disponibles",
        "cuales estan disponibles",
        "cuales estan en stock",
        "que esten disponibles",
        "que tengan stock",
        "en stock",
        "a la venta",
        "en venta",
    ]
    return any(pattern in normalized for pattern in patterns)


def _is_general_promotion_query(message: str) -> bool:
    normalized = _normalize_text(message)
    promotion_terms = ["promocion", "promociones", "oferta", "ofertas", "descuento", "descuentos", "rebaja", "rebajas"]
    if not any(term in normalized for term in promotion_terms):
        return False

    general_terms = ["en general", "generales", "todas", "tienen", "hay", "activas", "vigentes"]
    catalog_terms = get_catalog_reference_terms()
    has_catalog_reference = any(term in normalized for term in catalog_terms)
    has_general_hint = any(term in normalized for term in general_terms)
    return (has_general_hint and not has_catalog_reference) or normalized in {"promociones", "que promociones tienen", "que descuentos tienen"}


def _wants_catalog_categories(message: str) -> bool:
    normalized = _normalize_text(message)
    patterns = [
        "que categorias hay",
        "que categorias manejan",
        "que categorias tienen",
        "cuales categorias hay",
        "cuales categorias tienen",
        "categorias disponibles",
    ]
    return any(pattern in normalized for pattern in patterns)


def _is_general_confirmation(message: str) -> bool:
    normalized = _normalize_text(message)
    return normalized in {"en general", "general", "todas", "todas las promociones", "las generales", "todas en general"}


def _is_promotion_followup_query(message: str) -> bool:
    normalized = _normalize_text(message)
    if any(term in normalized for term in ["promocion", "promociones", "descuento", "descuentos", "oferta", "ofertas", "rebaja", "rebajas"]):
        return True

    short_followups = {
        "electrodomesticos",
        "televisores",
        "celulares",
        "laptops",
        "hogar",
        "deportes",
        "ropa",
        "calzado",
        "tecnologia",
        "electronica",
        "para electrodomesticos",
        "para televisores",
        "para celulares",
        "para laptops",
    }
    return normalized in short_followups


def _is_returns_policy_followup(message: str) -> bool:
    normalized = _normalize_text(message)
    followup_terms = [
        "reembolso",
        "reembolsos",
        "cambio",
        "cambios",
        "credito",
        "credito en tienda",
        "saldo a favor",
        "cual me conviene",
        "que me conviene",
        "cuál me conviene",
        "qué me conviene",
    ]
    return any(term in normalized for term in followup_terms)


def _is_returns_policy_query(message: str) -> bool:
    normalized = _normalize_text(message)
    return any(term in normalized for term in [
        "devolver",
        "devolucion",
        "devolución",
        "reembolso",
        "reembolsar",
        "cambio",
        "cambiar",
    ])


def _format_returns_policy_followup_response(message: str) -> str:
    normalized = _normalize_text(message)

    if "reembolso" in normalized:
        return (
            "Si prefieres reembolso, te devuelven el dinero al medio de pago original. "
            "Ten presente que el envío original no suele ser reembolsable y, según la política, "
            "podría aplicar un cargo administrativo en algunos casos. "
            "Si quieres, también te explico cuánto puede tardar en verse reflejado."
        )

    if "credito" in normalized or "saldo a favor" in normalized:
        return (
            "Si eliges crédito en tienda, el valor te queda como saldo a favor para otra compra. "
            "Suele ser la opción más ágil porque queda disponible apenas procesan la devolución. "
            "Te conviene más si piensas volver a comprar pero todavía no tienes claro qué escoger."
        )

    if "cambio" in normalized:
        return (
            "Si lo que quieres es cambio, puedes pedir otra referencia, talla, color o modelo, "
            "siempre que haya disponibilidad cuando revisen la devolución. "
            "Esa opción te conviene más si todavía quieres quedarte con un producto parecido."
        )

    return (
        "Depende de lo que prefieras: si quieres tu dinero de vuelta, te conviene reembolso; "
        "si planeas comprar otra cosa más adelante, crédito en tienda; "
        "y si quieres reemplazar ese producto por otro, cambio."
    )


def _is_product_followup_query(message: str) -> bool:
    normalized = _normalize_text(message)
    if _is_context_recall_message(message):
        return False

    if _is_product_recommendation_query(message):
        return True

    if _extract_contextual_brand_mentions(message):
        brand_interest_signals = [
            "interesado",
            "interesada",
            "me interesa",
            "quiero",
            "quisiera",
            "busco",
            "estoy interesada",
            "estoy interesado",
            "muéstrame",
            "muestrame",
            "tienes",
            "tienen",
            "hay",
            "disponible",
            "disponibles",
            "unos",
            "unas",
        ]
        if any(signal in normalized for signal in brand_interest_signals):
            return True

    if not any(term in normalized for term in get_catalog_reference_terms()):
        return False

    if normalized in set(get_catalog_reference_terms()):
        return True

    product_signals = [
        "interesado",
        "interesada",
        "me interesa",
        "tienes",
        "tienen",
        "hay",
        "manejan",
        "venden",
        "disponible",
        "disponibles",
        "stock",
        "busco",
        "quiero",
        "muestrame",
        "muéstrame",
        "unos",
        "unas",
    ]
    return any(signal in normalized for signal in product_signals)


def _intent_from_order_summary_followup(message: str) -> Optional[str]:
    normalized = _normalize_text(message)

    if _is_warranty_or_policy_topic(message):
        return None

    if any(term in normalized for term in ["total", "cuanto pague", "cuánto pagué", "subtotal", "iva"]):
        return "order_amount"

    if any(term in normalized for term in ["que pedi", "que ordene", "que compre", "productos", "articulos", "items", "incluia"]):
        return "order_items"

    if any(term in normalized for term in [
        "metodo de pago", "método de pago",
        "como pague", "cómo pagué", "como lo pague",
        "con que pague", "con qué pagué",
        "medio de pago", "forma de pago"
    ]):
        return "order_payment_method"

    if any(term in normalized for term in ["direccion", "dirección"]):
        return "order_delivery_address"

    if any(term in normalized for term in ["historial", "seguimiento", "tracking", "estado"]):
        return "order_status_history"

    return None


def _intent_from_contextual_order_followup(message: str, last_order_id: Optional[str]) -> Optional[str]:
    if last_order_id is None:
        return None

    normalized = _normalize_text(message)
    if not normalized:
        return None

    followup_intent = _intent_from_order_summary_followup(message)
    if followup_intent is not None:
        return followup_intent

    short_followup_patterns = {
        "order_amount": {"y el total", "el total", "cuanto pague", "y cuanto pague"},
        "order_items": {"que pedi", "que ordene", "que compre", "los productos", "los articulos", "el contenido"},
        "order_payment_method": {
            "y el metodo de pago", "el metodo de pago", "y como pague", "como pague",
            "como lo pague", "cual fue mi metodo de pago"
        },
        "order_status_history": {
            "y el estado", "el estado", "y el historial", "el historial",
            "seguimiento", "tracking", "historial completo", "me regalas el historial completo"
        },
        "order_delivery_address": {"y la direccion", "la direccion", "esa direccion", "a donde lo enviaron", "a donde iba"},
    }

    for intent, patterns in short_followup_patterns.items():
        if normalized in patterns:
            return intent

    return None


def _wants_full_order_history(message: str) -> bool:
    normalized = _normalize_text(message)
    detailed_markers = [
        "historial",
        "historial completo",
        "seguimiento",
        "tracking",
        "trazabilidad",
        "detalle completo",
        "detalles del envio",
        "detalles de envio",
        "guia",
        "numero de guia",
        "transportadora",
        "donde va",
        "por donde va",
        "me regalas el historial completo",
    ]
    return any(marker in normalized for marker in detailed_markers)


def _is_warranty_or_policy_topic(message: str) -> bool:
    """
    Detecta si el mensaje es sobre políticas o garantías.
    Usa raíces de palabras para cubrir conjugaciones y variantes
    (devolver, devolución, reembolsar, garantiza, etc.)
    """
    normalized = _normalize_text(message)

    # Términos exactos
    exact_markers = [
        "garantia", "garantias", "devolucion", "devoluciones",
        "reembolso", "reembolsos", "politica", "politicas",
        "cambio", "cambios",
    ]
    if any(marker in normalized for marker in exact_markers):
        return True

    # Raíces de palabras (cubre conjugaciones)
    policy_stems = ["devol", "reembols", "garantiz", "cancelar", "cancelaci"]
    if any(stem in normalized for stem in policy_stems):
        return True

    return False


def _select_order_from_summary(selection_result: dict, message: str) -> Optional[str]:
    if not selection_result.get("success"):
        return None

    results = selection_result.get("results") or []
    if not results:
        return None

    explicit_order_id = _extract_order_id(message)
    if explicit_order_id is not None:
        for item in results:
            item_order_id = str(item.get("order_id"))
            if item_order_id.upper() == explicit_order_id.upper():
                return item_order_id

    if _wants_last_order(message):
        return str(results[0]["order_id"])

    desired_status = _wants_order_by_status(message)
    if desired_status:
        for item in results:
            if item["status"] == desired_status:
                return str(item["order_id"])

    return None


def _humanize_order_status(status: Optional[str]) -> str:
    mapping = {
        "delivered": "entregado",
        "cancelled": "cancelado",
        "shipped": "enviado",
        "preparing": "en preparación",
        "payment_confirmed": "pago confirmado",
        "ready_for_pickup": "listo para recogida",
        "in_transit": "en camino",
        "out_for_delivery": "en reparto",
        "order_placed": "pedido creado",
        "returned": "devuelto",
    }
    normalized = _normalize_text(status or "")
    return mapping.get(normalized, status or "sin estado")


def _humanize_delivery_method(method: Optional[str]) -> str:
    mapping = {
        "home_delivery": "envío a domicilio",
        "pickup_point": "recogida en punto",
        "store_pickup": "recogida en tienda",
    }
    normalized = _normalize_text(method or "")
    return mapping.get(normalized, (method or "no disponible").replace("_", " "))


def _auth_prompt(topic: str = "esa consulta") -> str:
    return (
        f"Te ayudo con {topic}. "
        "Antes necesito confirmar que la cuenta sí es tuya, "
        "así que compárteme tu cédula o el teléfono registrado y sigo con eso."
    )


def _order_number_prompt() -> str:
    return (
        "Listo. Si me compartes el número del pedido, lo reviso enseguida. "
        "Y si no lo recuerdas, también puedo mostrarte tus pedidos recientes."
    )


def _order_followup_prompt() -> str:
    return (
        "Si quieres, también puedo decirte qué productos venían, "
        "cuánto pagaste, cómo lo pagaste, a qué dirección iba "
        "o mostrarte el historial completo."
    )


def _format_order_status_response(result: dict) -> str:
    if not result["success"]:
        return result["message"]

    lines = [
        f"Ya revisé tu pedido {result['order_id']}.",
        f"En este momento aparece como {_humanize_order_status(result['status'])}.",
    ]

    if result.get("order_date"):
        lines.append(f"Fecha del pedido: {result['order_date']}.")
    if result.get("shipped_at"):
        lines.append(f"Fecha de despacho: {result['shipped_at']}.")
    if result.get("delivered_at"):
        lines.append(f"Fecha de entrega: {result['delivered_at']}.")
    if result.get("cancelled_at"):
        lines.append(f"Fecha de cancelación: {result['cancelled_at']}.")
    if result.get("delivery_method"):
        lines.append(f"Método de entrega: {_humanize_delivery_method(result['delivery_method'])}.")

    return "\n".join(lines)


def _format_order_summary_options(
    status_result: dict,
    shipment_result: Optional[dict] = None
) -> str:
    if not status_result["success"]:
        return status_result["message"]

    lines = [
        f"Ya revisé tu pedido {status_result['order_id']}.",
        f"Por ahora aparece como {_humanize_order_status(status_result['status'])}.",
    ]

    if status_result.get("order_date"):
        lines.append(f"Lo hiciste el {status_result['order_date']}.")

    if shipment_result and shipment_result.get("success") and shipment_result.get("estimated_delivery"):
        lines.append(f"La fecha estimada de entrega es {shipment_result['estimated_delivery']}.")

    lines.append(_order_followup_prompt())
    return "\n".join(lines)


def _format_order_amounts_response(result: dict) -> str:
    if not result["success"]:
        return result["message"]

    def _format_money(value: float) -> str:
        return f"${value:,.0f}".replace(",", ".")

    return (
        f"Claro, este es el resumen de valores de tu pedido {result['order_id']}:\n\n"
        f"Subtotal: {_format_money(result['subtotal'])}\n"
        f"IVA: {_format_money(result['tax'])}\n"
        f"Costo de envío: {_format_money(result['shipping_cost'])}\n"
        f"Total pagado: {_format_money(result['total_amount'])}\n\n"
        f"En este momento el pedido aparece como {_humanize_order_status(result['status'])}."
    )


def _format_order_items_response(result: dict) -> str:
    if not result["success"]:
        return result["message"]

    def _format_money(value: float) -> str:
        return f"${value:,.0f}".replace(",", ".")

    lines = [f"Esto fue lo que encontré en tu pedido {result['order_id']}:"]

    for item in result["items"]:
        line = (
            f"- {item['product_name']} x{item['qty']} "
            f"({_format_money(item['unit_price'])} c/u, subtotal {_format_money(item['line_total'])})"
        )
        if item.get("item_status") and item["item_status"] != "active":
            line += f" [{item['item_status']}]"
        lines.append(line)

    return "\n".join(lines)


def _format_payment_method_response(result: dict) -> str:
    if not result["success"]:
        return result["message"]

    payment_method = result["payment_method"] or "No disponible"
    payment_method = payment_method.replace("_", " ")
    lines = [
        f"Tu pedido {result['order_id']} aparece registrado con este método de pago: {payment_method}."
    ]

    if result.get("card"):
        card = result["card"]
        lines.append(
            f"Tarjeta: {card['card_type']} del banco {card['bank']} terminada en {card['last_four']}."
        )

    if result.get("payment_confirmed_at"):
        lines.append(f"Pago confirmado el: {result['payment_confirmed_at']}.")

    return "\n".join(lines)


def _format_customer_address_response(result: dict) -> str:
    if not result["success"]:
        return result["message"]

    lines = [
        "Esta es la dirección principal que tienes registrada:",
        result["address_line1"],
    ]

    if result.get("address_line2"):
        lines.append(result["address_line2"])

    lines.append(f"{result['city']}, {result['department']}, {result['postal_code']}")
    lines.append(result["country"])
    lines.append(f"Tipo de dirección: {result['address_type']}.")

    if result.get("landmark"):
        lines.append(f"Referencia: {result['landmark']}.")

    return "\n".join(lines)


def _format_order_delivery_address_response(result: dict) -> str:
    if not result["success"]:
        return result["message"]

    lines = [
        f"Esta es la dirección de entrega que aparece para tu pedido {result['order_id']}:",
        result["address_line1"],
    ]

    if result.get("address_line2"):
        lines.append(result["address_line2"])

    lines.append(f"{result['city']}, {result['department']}, {result['postal_code']}")
    lines.append(result["country"])

    if result.get("delivery_method"):
        lines.append(f"Método de entrega: {result['delivery_method']}.")

    if result.get("landmark"):
        lines.append(f"Referencia: {result['landmark']}.")

    return "\n".join(lines)


def _format_order_history_response(
    history_result: dict,
    shipment_result: dict,
    status_result: Optional[dict] = None
) -> str:
    error_messages = [
        result.get("message")
        for result in [status_result, history_result, shipment_result]
        if result is not None and not result.get("success") and result.get("message")
    ]
    if error_messages:
        unique_messages = list(dict.fromkeys(error_messages))
        if len(unique_messages) == 1:
            return unique_messages[0]

    if (
        (status_result is None or not status_result["success"])
        and not history_result["success"]
        and not shipment_result["success"]
    ):
        return "No encontré historial ni información logística para ese pedido."

    def _format_tracking_timestamp(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        raw = str(value).strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(raw, fmt)
                if fmt == "%Y-%m-%d":
                    return dt.strftime("%d/%m/%Y")
                return dt.strftime("%d/%m/%Y a las %H:%M")
            except ValueError:
                continue
        return raw

    def _humanize_tracking_status(status: Optional[str]) -> str:
        mapping = {
            "order_placed": "se registró el pedido",
            "payment_confirmed": "se confirmó el pago",
            "preparing": "empezaron a prepararlo",
            "ready_for_pickup": "quedó listo para despacho",
            "shipped": "salió a transporte",
            "in_transit": "iba en tránsito",
            "out_for_delivery": "salió a entrega",
            "delivered": "se entregó",
            "cancelled": "se canceló",
            "returned": "se devolvió",
        }
        normalized_status = str(status or "").strip().lower()
        if normalized_status in mapping:
            return mapping[normalized_status]
        return normalized_status.replace("_", " ") or "hubo una actualización"

    def _humanize_logistics_status(status: Optional[str]) -> str:
        mapping = {
            "delivered": "entregado",
            "in_transit": "en tránsito",
            "out_for_delivery": "en reparto",
            "shipped": "despachado",
            "returned": "devuelto",
            "cancelled": "cancelado",
        }
        normalized_status = str(status or "").strip().lower()
        if normalized_status in mapping:
            return mapping[normalized_status]
        return normalized_status.replace("_", " ") or "sin estado visible"

    lines = []

    if status_result is not None and status_result["success"]:
        opening = (
            f"Claro, te cuento el recorrido completo de tu pedido {status_result['order_id']}."
        )
        lines.append(opening)
        lines.append(f"En este momento aparece como {_humanize_order_status(status_result['status'])}.")
        if status_result.get("delivery_method"):
            lines.append(
                f"La entrega fue por {_humanize_delivery_method(status_result['delivery_method']).lower()}."
            )

    if shipment_result["success"]:
        shipment_bits = []
        if shipment_result.get("carrier"):
            shipment_bits.append(f"la transportadora fue {shipment_result['carrier']}")
        if shipment_result.get("tracking_number"):
            shipment_bits.append(f"la guía es {shipment_result['tracking_number']}")
        if shipment_result.get("status"):
            shipment_bits.append(
                f"el estado logístico figura como {_humanize_logistics_status(shipment_result['status'])}"
            )
        if shipment_bits:
            lines.append("Además, veo que " + ", ".join(shipment_bits) + ".")
        if shipment_result.get("estimated_delivery"):
            lines.append(f"La entrega estimada estaba para {shipment_result['estimated_delivery']}.")
        if shipment_result.get("actual_delivery"):
            lines.append(f"Y la entrega real quedó registrada el {shipment_result['actual_delivery']}.")
        attempts = shipment_result.get("delivery_attempts", 0)
        if attempts == 1:
            lines.append("Se registró 1 intento de entrega.")
        else:
            lines.append(f"Se registraron {attempts} intentos de entrega.")
        if shipment_result.get("failed_reason"):
            lines.append(f"También aparece esta novedad logística: {shipment_result['failed_reason']}.")
        if shipment_result.get("tracking_url"):
            lines.append(f"Si quieres hacer seguimiento por tu cuenta, puedes usar esta URL: {shipment_result['tracking_url']}")

    if history_result["success"]:
        lines.append("")
        lines.append("Te lo resumo en orden:")
        for event in history_result["history"]:
            when = _format_tracking_timestamp(event.get("timestamp")) or "en una fecha no registrada"
            happened = _humanize_tracking_status(event.get("status"))
            location = event["location"] if event.get("location") else None

            if location and location.lower() != "dirección de entrega":
                lines.append(f"- El {when}, {happened} en {location}.")
            elif location and location.lower() == "dirección de entrega":
                lines.append(f"- El {when}, {happened} en la dirección de entrega.")
            else:
                lines.append(f"- El {when}, {happened}.")

    return "\n".join(lines).strip()


def _format_returns_case_response(
    history_result: dict,
    shipment_result: dict,
    status_result: Optional[dict] = None
) -> str:
    if status_result is not None and status_result["success"]:
        if status_result["status"] == "returned":
            return _format_order_history_response(history_result, shipment_result, status_result)

        return (
            f"Revisé tu pedido {status_result['order_id']} y por ahora no aparece como devuelto. "
            f"El estado actual es {_humanize_order_status(status_result['status'])}. "
            "Si quieres, te puedo mostrar el historial completo del pedido."
        )

    return _format_order_history_response(history_result, shipment_result, status_result)


def _format_warranty_response(result: dict) -> str:
    if not result["success"]:
        return result["message"]

    lines = ["Claro, esto es lo que encontré sobre garantía en esa categoría o esos productos:"]
    for item in result.get("results", []):
        warranty_months = item.get("warranty_months", 0)
        return_days = item.get("return_days", 0)
        category = item.get("category") or "sin categoría"
        lines.append(
            f"- {item['name']} ({category}): garantía de {warranty_months} meses y devolución de {return_days} días."
        )

    lines.append("Si quieres, también te lo resumo por producto o por categoría.")
    return "\n".join(lines)


def _format_product_response(result: dict) -> str:
    if not result["success"]:
        return result["message"]

    def _format_price(value: float) -> str:
        return f"${value:,.0f}".replace(",", ".")

    def _detect_collection_label(query: str, results: list) -> str:
        normalized = _normalize_text(query)
        collection_aliases = {
            "ropa": ["ropa", "camiseta", "camisetas", "jeans", "chaqueta", "chaquetas", "vestido", "vestidos", "sudadera", "sudaderas"],
            "calzado": ["calzado", "zapato", "zapatos", "zapatilla", "zapatillas", "tenis", "sneaker", "sneakers"],
            "celulares": ["celulares", "celular", "telefonos", "telefono"],
            "televisores": ["televisores", "televisor", "tv"],
            "electrodomesticos": ["electrodomesticos", "electrodomestico", "neveras", "lavadoras", "estufas", "microondas", "cafeteras", "licuadoras"],
            "electronica": ["electronica", "monitores", "audifonos", "tablets"],
            "hogar y muebles": ["hogar", "muebles", "sofas", "camas", "mesas", "escritorios", "colchones"],
            "deportes y fitness": ["deportes", "fitness", "bicicletas", "bicicleta", "trotadora", "trotadoras", "mancuernas", "balones", "balon", "raquetas", "raqueta", "yoga"],
            "belleza y cuidado personal": ["belleza", "cuidado personal", "secador", "secadores", "plancha", "planchas", "perfume", "perfumes", "skincare", "afeitar", "maquillaje"],
            "libros y papeleria": ["libros", "libro", "papeleria", "agenda", "agendas", "marcadores", "cuaderno", "cuadernos", "kindle"],
            "juguetes y bebes": ["juguetes", "juguete", "bebe", "bebes", "lego", "muneca", "munecas", "panales", "coche", "triciclo"],
            "laptops": ["laptops", "laptop", "portatiles", "portatil"],
        }
        for label, aliases in collection_aliases.items():
            if any(alias in normalized for alias in aliases):
                return label

        first_category = results[0].get("category") if results else None
        return first_category.lower() if first_category else "productos"

    collection_label = _detect_collection_label(result.get("query", ""), result["results"])

    # Variación de intros para evitar respuestas repetitivas
    intros = [
        f"Esto es lo que tenemos en {collection_label}:",
        f"Acá las opciones de {collection_label} que encontré:",
        f"Encontré estas opciones en {collection_label}:",
        f"Mira, esto hay en {collection_label}:",
        f"Te cuento lo que hay en {collection_label}:",
    ]
    lines = [random.choice(intros)]

    for item in result["results"]:
        details = [f"queda en {_format_price(item['price'])}"]
        if item["available_qty"] > 0:
            details.append(f"hay {item['available_qty']} disponibles")
        else:
            details.append("ahora mismo no tiene stock")

        if item["free_shipping"]:
            details.append("tiene envío gratis")

        if item.get("shipping_days"):
            details.append(f"entrega estimada en {item['shipping_days']} días")

        if item.get("is_final_sale"):
            details.append("aplica como venta final")

        if item["requires_installation"]:
            details.append("requiere instalación")

        if not item["active"]:
            details.append("esta referencia aparece inactiva")

        brand_prefix = f"{item['brand']} " if item.get("brand") and item["brand"].lower() not in item["name"].lower() else ""
        lines.append(f"- {brand_prefix}{item['name']}: {', '.join(details)}.")

    lines.append("")
    lines.append("Si quieres, te los filtro por precio, marca o disponibilidad.")
    return "\n".join(lines)


def _format_purchase_guidance_response(result: dict) -> str:
    if not result["success"] or not result.get("results"):
        return (
            "Puedo orientarte con la compra, pero primero necesitaría confirmar el producto exacto "
            "que te interesa."
        )

    item = result["results"][0]
    lines = [
        f"Si quieres comprar el {item['name']}, puedes agregarlo al carrito y completar el pago en el checkout.",
    ]

    if item["available_qty"] > 0:
        lines.append(f"Ahora mismo hay {item['available_qty']} unidades disponibles.")
    else:
        lines.append("Ahora mismo no veo stock disponible para esa referencia.")

    lines.append(f"Precio actual: ${item['price']:,.0f}".replace(",", ".") + ".")

    if item["free_shipping"]:
        lines.append("Esta referencia tiene envío gratis.")
    if item.get("shipping_days"):
        lines.append(f"La entrega estimada es de {item['shipping_days']} días.")

    lines.append(
        "Si quieres, también te puedo contar si tiene promociones activas o mostrarte opciones similares."
    )
    return "\n".join(lines)


def _format_promotion_response(result: dict) -> str:
    if not result["success"]:
        return result["message"]

    def _format_money(value: float) -> str:
        return f"${value:,.0f}".replace(",", ".")

    # Variación de intros para evitar respuestas repetitivas
    intros = [
        "Estas son las promociones activas que encontré:",
        "Mira las promociones que tenemos ahora mismo:",
        "Acá las promociones vigentes:",
        "Te cuento las promociones activas:",
        "Encontré estas promociones:",
    ]
    lines = [random.choice(intros)]

    for item in result["results"]:
        if item["discount_type"] == "percentage":
            benefit = f"te da {int(item['discount_value'])}% de descuento"
        elif item["discount_type"] == "fixed_amount":
            benefit = f"te descuenta {_format_money(item['discount_value'])}"
        elif item["discount_type"] == "free_shipping":
            benefit = "te deja el envío gratis"
        else:
            benefit = item["description"]

        target_text = ""
        if item["targets"]:
            target_text = f" en {', '.join(item['targets'][:3])}"

        minimum_text = ""
        if item["min_purchase_amount"] > 0:
            minimum_text = f" y aplica desde compras de {_format_money(item['min_purchase_amount'])}"

        validity_text = f" La encuentras vigente hasta {item['end_date']}."
        lines.append(f"- {item['promotion_name']}: {benefit}{target_text}{minimum_text}.{validity_text}")

    lines.append("")
    lines.append("Si quieres, te digo cuáles te convienen más para una categoría o un producto específico.")
    return "\n".join(lines)


def _promotion_clarification_response() -> str:
    return (
        "Claro. Te puedo mostrar las promociones generales o las que aplican a una categoría o producto en específico.\n"
        "Si quieres, dime por ejemplo celulares, televisores o una marca. Y si no, también te las muestro en general."
    )


def _format_categories_response(result: dict) -> str:
    if not result["success"]:
        return result["message"]

    lines = ["Claro, estas son las categorías que maneja la tienda:"]
    for item in result["results"]:
        lines.append(f"- {item['name']} ({item['product_count']} productos en catálogo)")

    lines.append("")
    lines.append("Si quieres, te muestro opciones de alguna en particular.")
    return "\n".join(lines)


def _user_does_not_remember_order_id(message: str) -> bool:
    msg = _normalize_text(message)
    patterns = [
        "no recuerdo",
        "no lo recuerdo",
        "no me acuerdo",
        "no lo se",
        "no se el numero",
        "no se cual es",
        "no tengo el numero",
        "muestrame mis pedidos",
        "cuales son mis pedidos",
        "mis pedidos",
        "no recuerdo el numero de pedido",
    ]
    return any(p in msg for p in patterns)


def _format_customer_orders_summary(result: dict) -> str:
    if not result["success"]:
        return (
            "Ya validé tu cuenta, pero no encontré pedidos asociados. "
            "Si crees que esto es un error, revisa si compartiste el dato correcto de cédula o teléfono."
        )

    def _format_total(value: float) -> str:
        return f"${value:,.0f}".replace(",", ".")

    def _friendly_status(status: str) -> str:
        mapping = {
            "delivered": "entregado",
            "cancelled": "cancelado",
            "shipped": "enviado",
            "processing": "en preparación",
            "returned": "devuelto",
        }
        return mapping.get(str(status).lower(), str(status))

    # Variación de intros para evitar la frase repetida "Claro, encontré estos pedidos recientes"
    intros = [
        "Estos son tus pedidos recientes:",
        "Acá están tus últimos pedidos:",
        "Te muestro tus pedidos más recientes:",
        "Revisé tu cuenta y encontré estos pedidos:",
        "Mira, estos son tus pedidos:",
    ]
    lines = [random.choice(intros)]

    for item in result["results"]:
        order_date = str(item["order_date"]).split(" ")[0] if item.get("order_date") else "sin fecha visible"
        lines.append(
            f"- El pedido {item['order_id']} fue {_friendly_status(item['status'])}, "
            f"lo hiciste el {order_date} y quedó por {_format_total(item['total_amount'])}."
        )

    lines.append("")
    lines.append("Dime cuál quieres revisar y lo miro contigo.")
    return "\n".join(lines)


def _build_intent_hint(
    message: str,
    session_customer=None,
    pending_intent: Optional[str] = None,
    pending_offer: Optional[str] = None,
):
    return _classify_intent(
        message,
        session_customer=session_customer,
        pending_intent=pending_intent,
        pending_offer=pending_offer,
    )


def _build_hybrid_system_prompt(snapshot: dict, runtime_metadata: dict) -> str:
    session_customer = snapshot.get("session_customer")
    conversation_state = snapshot.get("conversation_state", {})

    # Sección de autenticación
    if session_customer:
        auth_section = (
            f"El cliente está autenticado. "
            f"customer_id={session_customer['customer_id']}, "
            f"nombre={session_customer.get('display_name', 'cliente')}."
        )
    else:
        auth_section = "El cliente NO está autenticado todavía."

    # Contexto de conversación en lenguaje natural, no como key=value
    context_parts = []

    last_order_id = conversation_state.get("last_order_id")
    if last_order_id:
        context_parts.append(
            f"El cliente acaba de consultar el pedido {last_order_id}. "
            f"Si menciona 'ese pedido', 'el pedido', 'cómo lo pagué', 'y la dirección', "
            f"'historial completo' u otro followup sin especificar número, "
            f"siempre se refiere al pedido {last_order_id}."
        )

    recent_order_ids = conversation_state.get("recent_order_ids") or []
    if recent_order_ids and not last_order_id:
        context_parts.append(f"Pedidos recientes vistos en esta sesión: {recent_order_ids}.")

    last_product_query = conversation_state.get("last_product_query")
    if last_product_query:
        context_parts.append(
            f"El cliente preguntó sobre '{last_product_query}' antes. "
            f"Si dice 'esos', 'ese producto', 'de esos' o similar, se refiere a eso."
        )

    pending_intent = conversation_state.get("pending_intent")
    if pending_intent:
        context_parts.append(
            f"El cliente estaba preguntando sobre '{pending_intent}' pero aún no se resolvió. "
            f"Si provee datos de autenticación o el número de pedido, continúa con esa consulta."
        )

    pending_offer = conversation_state.get("pending_offer")
    if pending_offer == "order_summary_options":
        context_parts.append(
            "Se le ofreció al cliente ver más detalles del pedido. "
            "Si responde afirmativamente o pide un detalle específico, dáselo."
        )

    context_section = "\n".join(context_parts) if context_parts else "No hay contexto previo relevante."

    # Tool trace resumido
    tool_trace_section = _summarize_recent_tool_trace(limit=4)

    return (
        "Eres un agente conversacional retail basado en IA generativa.\n"
        "Ayudas al usuario de forma natural, clara y útil.\n\n"
        "REGLAS DURAS:\n"
        "- Nunca inventes datos de pedidos, pagos, direcciones, promociones, stock o políticas.\n"
        "- Solo usa herramientas (tools) cuando necesites verificar datos reales o sensibles.\n"
        "- Si puedes responder con orientación general, una aclaración o una pregunta de precisión, hazlo sin tools.\n"
        "- Si la consulta requiere autenticación y el cliente no está autenticado, pide cédula o teléfono.\n"
        "- Para consultas sobre políticas de devolución, garantía o envío, SIEMPRE usa search_policy_sections.\n"
        "  No respondas políticas de memoria. Llama la tool primero.\n"
        "- search_policy_sections es para preguntas sobre cómo funcionan las políticas, plazos, condiciones.\n"
        "  NO la uses cuando el cliente quiere hacer una devolución de un pedido concreto suyo\n"
        "  (en ese caso usa get_order_history + get_order_status).\n"
        "- Para followups como 'cómo lo pagué', 'y la dirección', 'historial completo', 'el total':\n"
        "  usa el pedido del contexto directamente, sin pedir número de nuevo.\n"
        "- No llames una tool solo porque exista una intención sugerida; decide si realmente necesitas evidencia.\n"
        "- No repitas la misma frase introductoria si ya la usaste en el turno anterior.\n"
        "- No suenes como consola ni como sistema interno.\n"
        "- No compartas número completo de tarjeta, CVV ni datos sensibles.\n\n"
        "ESTILO:\n"
        "- Responde en español colombiano, conversacional y natural.\n"
        "- Sé breve y resolutivo. El cliente quiere respuestas, no explicaciones largas.\n"
        "- Varía las frases de introducción en cada respuesta.\n"
        "- Cuando uses datos de tools, intégralos naturalmente en la respuesta.\n"
        "- Si el usuario está explorando opciones o todavía no dio suficiente contexto, primero aclara o guía sin consultar.\n\n"
        f"ESTADO DE SESIÓN:\n{auth_section}\n\n"
        f"CONTEXTO DE CONVERSACIÓN:\n{context_section}\n\n"
        f"TOOLS USADAS RECIENTEMENTE:\n{tool_trace_section}\n\n"
        f"Runtime LLM: provider={runtime_metadata.get('provider')}, model={runtime_metadata.get('model')}."
    )


def _tool_result_to_content(name: str, result: dict) -> str:
    return json.dumps({
        "tool_name": name,
        "result": result,
    }, ensure_ascii=False)


def _with_runtime_marker(response: str) -> str:
    debug_enabled = os.getenv("AGENT_DEBUG_RESPONSE_SOURCE", "false").strip().lower() in {"1", "true", "yes", "on"}
    if not debug_enabled:
        return response

    source = get_last_response_source() or "unknown"
    llm_error = get_last_llm_error()
    marker = f"[debug] response_source={source}"
    if llm_error:
        marker += f" | llm_error={llm_error}"
    return f"{response}\n\n{marker}"


def _llm_strict_mode_enabled() -> bool:
    strict_flags = [
        "AGENT_STRICT_LLM_ONLY",
        "AGENT_REQUIRE_LLM",
        "AGENT_EVAL_REQUIRE_LLM",
    ]
    return any(
        os.getenv(flag, "false").strip().lower() in {"1", "true", "yes", "on"}
        for flag in strict_flags
    )


def _build_llm_unavailable_response(llm_client, runtime_metadata: dict) -> str:
    provider = runtime_metadata.get("provider") or getattr(llm_client, "provider_name", "llm")
    model = runtime_metadata.get("model")
    reason = None
    if hasattr(llm_client, "get_unavailable_reason"):
        reason = llm_client.get_unavailable_reason()

    detail = f"Proveedor configurado: {provider}"
    if model:
        detail += f", modelo: {model}"

    if reason:
        return (
            "El modo LLM está marcado como obligatorio, pero no pude inicializar el proveedor configurado. "
            f"{detail}. Motivo: {reason}"
        )

    return (
        "El modo LLM está marcado como obligatorio, pero no pude inicializar el proveedor configurado. "
        f"{detail}."
    )


def _summarize_recent_tool_trace(limit: int = 4) -> str:
    traces = get_recent_tool_trace(limit=limit)
    if not traces:
        return "No hay tools recientes en memoria."

    summaries = []
    for trace in traces:
        tool_name = trace.get("tool_name")
        output = trace.get("output")
        if not isinstance(output, dict):
            summaries.append(f"{tool_name}: resultado no estructurado.")
            continue

        if tool_name in {
            "get_order_status", "get_order_amounts", "get_order_items",
            "get_order_payment_method", "get_order_history",
            "get_shipment_details", "get_order_delivery_address"
        }:
            order_id = output.get("order_id")
            success = output.get("success")
            summaries.append(f"{tool_name}: success={success}, order_id={order_id}.")
            continue

        if tool_name in {"get_customer_orders_summary", "get_customer_orders_for_selection"}:
            results = output.get("results") or []
            order_ids = [item.get("order_id") for item in results if isinstance(item, dict)]
            summaries.append(f"{tool_name}: pedidos recientes={order_ids}.")
            continue

        if tool_name == "search_products":
            results = output.get("results") or []
            product_names = [item.get("name") for item in results[:3] if isinstance(item, dict)]
            summaries.append(f"{tool_name}: productos={product_names}.")
            continue

        if tool_name == "search_promotions":
            results = output.get("results") or []
            promo_names = [item.get("promotion_name") for item in results[:3] if isinstance(item, dict)]
            summaries.append(f"{tool_name}: promociones={promo_names}.")
            continue

        if tool_name == "search_policy_sections":
            results = output.get("results") or []
            titles = [item.get("title") or item.get("header") for item in results[:2] if isinstance(item, dict)]
            summaries.append(f"{tool_name}: secciones={titles}.")
            continue

        summaries.append(f"{tool_name}: success={output.get('success')}.")

    return "\n".join(summaries)


def _update_session_state_from_tool_result(tool_name: str, arguments: dict, result: dict):
    if not isinstance(result, dict):
        return

    if tool_name == "authenticate_customer" and result.get("success"):
        return

    if tool_name in {
        "get_order_status",
        "get_order_amounts",
        "get_order_items",
        "get_order_payment_method",
        "get_order_history",
        "get_shipment_details",
        "get_order_delivery_address",
    } and result.get("success"):
        order_id = result.get("order_id") or arguments.get("order_id")
        if order_id is not None:
            order_id = str(order_id)
            set_last_order_id(order_id)
            set_recent_order_ids([order_id])
        return

    if tool_name in {"get_customer_orders_summary", "get_customer_orders_for_selection"} and result.get("success"):
        results = result.get("results") or []
        order_ids = [str(item.get("order_id")) for item in results if isinstance(item, dict) and item.get("order_id") is not None]
        if order_ids:
            set_recent_order_ids(order_ids)
        return

    if tool_name == "search_products" and arguments.get("query"):
        set_last_product_query(str(arguments["query"]))
        set_pending_offer("product_filters")
        brands, categories = _extract_topic_brands_and_categories(result)
        _remember_product_topic(str(arguments["query"]), brands=brands, categories=categories)
        return

    if tool_name == "search_promotions":
        set_pending_offer("promotion_followup")
        _remember_promotion_topic(str(arguments.get("query", "")), _get_memory_product_query())
        return


def _update_session_state_from_final_response(
    intent_data: dict,
    trace_start_index: int,
    response_text: str,
):
    normalized = _normalize_text(response_text)
    traces = get_tool_trace_since(trace_start_index)
    intent = intent_data.get("intent")

    if traces:
        return

    if intent in {
        "order_amount",
        "order_items",
        "order_payment_method",
        "order_status_history",
        "returns_order_case",
        "customer_address",
        "order_delivery_address",
    }:
        if any(marker in normalized for marker in ["cedula", "cédula", "telefono", "teléfono", "identidad", "verificar tu identidad"]):
            set_pending_intent(intent)
            return
        if any(marker in normalized for marker in [
            "numero del pedido", "número del pedido",
            "numero de tu pedido", "número de tu pedido",
            "cual quieres revisar", "cuál quieres revisar"
        ]):
            set_pending_intent(intent)
            return

    if intent == "promotion_general" and any(marker in normalized for marker in [
        "promociones generales", "categoria o producto", "categoría o producto", "te puedo mostrar"
    ]):
        set_pending_offer("promotion_scope")
        return

    if intent == "policy" and (
        _is_returns_policy_query(response_text)
        or any(marker in normalized for marker in [
            "cambio, crédito o reembolso",
            "cambio, credito o reembolso",
            "te conviene más cambio",
            "te conviene mas cambio",
        ])
    ):
        set_pending_offer("returns_policy_followup")
        return

    if intent in {"faq", "greeting"}:
        _clear_pending_flow()


def _resolve_authenticated_followup(
    auth_result: dict,
    pending_intent: Optional[str],
    pending_offer: Optional[str],
    trace_start_index: int,
) -> str:
    if pending_offer == "resolve_recent_order_after_auth":
        intent_after_auth = pending_intent or "order_status_history"
        response = _resolve_recent_order_after_auth(intent_after_auth)
        return _validate_or_return(intent_after_auth, trace_start_index, response)

    if pending_offer == "show_recent_orders_after_auth":
        summary_result = (
            get_customer_orders_for_selection(limit=10)
            if pending_intent in ["order_amount", "order_items", "order_payment_method", "order_delivery_address"]
            else get_customer_orders_summary(limit=5)
        )
        set_pending_offer(None)
        return _validate_or_return(
            pending_intent if pending_intent in ["order_amount", "order_items", "order_payment_method", "order_delivery_address"] else "order_status_history",
            trace_start_index,
            _format_customer_orders_summary(summary_result)
        )

    if pending_intent == "customer_address":
        address_result = get_customer_default_address()
        _clear_pending_flow()
        return _validate_or_return(
            "customer_address",
            trace_start_index,
            _format_customer_address_response(address_result)
        )

    if pending_intent == "order_delivery_address":
        order_id = _get_memory_order_id()
        if order_id is not None:
            address_result = get_order_delivery_address(order_id)
            _clear_pending_flow()
            return _validate_or_return(
                "order_delivery_address",
                trace_start_index,
                _format_order_delivery_address_response(address_result)
            )

    if pending_intent in ["order_amount", "order_items", "order_payment_method", "order_status_history", "returns_order_case"]:
        remembered_order_id = _get_memory_order_id()
        if remembered_order_id is not None:
            response = _resolve_selected_order_intent(pending_intent, remembered_order_id)
            validation_intent = "order_status_history" if pending_intent == "order_status_history" else pending_intent
            if pending_intent == "order_status_history":
                set_pending_offer("order_summary_options")
            return _validate_or_return(validation_intent, trace_start_index, response)

        return (
            f"Perfecto, {auth_result['name']}. Ya confirmé tu cuenta.\n"
            f"{_order_number_prompt()}"
        )

    _clear_pending_flow()
    return (
        f"Perfecto, {auth_result['name']}. Ya confirmé tu cuenta. "
        "Ahora sí, cuéntame qué quieres revisar y lo vemos juntos."
    )


def _handle_identifier_auth_flow(identifier: str, trace_start_index: int) -> str:
    pending_intent = get_pending_intent()
    pending_offer = get_pending_offer()
    auth_result = authenticate_customer(identifier)

    if auth_result["success"]:
        return _resolve_authenticated_followup(auth_result, pending_intent, pending_offer, trace_start_index)

    return _format_auth_failure_response()


def _messages_to_responses_input(messages: list) -> list:
    response_items = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role in {"system", "user", "assistant"} and isinstance(content, str):
            response_items.append({
                "role": role,
                "content": [{"type": "input_text", "text": content}],
            })
            continue

        if role == "assistant" and message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                function = tool_call.get("function") or {}
                response_items.append({
                    "type": "function_call",
                    "call_id": tool_call.get("id"),
                    "name": function.get("name"),
                    "arguments": function.get("arguments", "{}"),
                })
            continue

        if role == "tool":
            response_items.append({
                "type": "function_call_output",
                "call_id": message.get("tool_call_id"),
                "output": content or "",
            })

    return response_items


def _select_tools_for_llm(message: str, intent_hint: dict) -> list[dict]:
    intent = intent_hint.get("intent")

    if intent in {"greeting", "faq"}:
        return []

    if intent == "promotion_general" and _is_general_promotion_query(message):
        return []

    if intent_hint.get("requires_auth") and get_session_customer() is None:
        return []

    return get_tool_definitions_for_llm()


def _resolve_topic_recall_with_memory(message: str, trace_start_index: int) -> Optional[str]:
    normalized = _normalize_text(message)
    recent_product_topic = get_recent_topic("product")
    recent_promotion_topic = get_recent_topic("promotion")
    recent_policy_topic = get_recent_topic("policy")
    recent_order_topic = get_recent_topic("order")

    if recent_policy_topic and _is_returns_policy_followup(message):
        set_pending_offer("returns_policy_followup")
        return _format_returns_policy_followup_response(message)

    if recent_promotion_topic and any(term in normalized for term in ["promocion", "promociones", "descuento", "descuentos", "oferta", "ofertas"]):
        product_query = (
            _get_recent_product_query_for_promotion_context()
            if _is_general_promotion_query(message)
            else recent_promotion_topic.get("data", {}).get("product_query") or _get_recent_product_query_for_promotion_context()
        )
        promotion_query = _build_contextual_promotion_query(message, product_query)
        promotion_result = _filter_promotion_result_by_context(
            search_promotions(promotion_query),
            promotion_query,
            product_query,
        )
        _remember_promotion_topic(promotion_query, product_query)
        set_pending_offer("promotion_followup")
        return _validate_or_return(
            "promotion_general",
            trace_start_index,
            _format_promotion_response(promotion_result),
        )

    if (
        recent_product_topic
        and not _is_promotion_followup_query(message)
        and not _is_general_promotion_query(message)
        and (_extract_brand_mentions(message) or _is_context_recall_message(message))
    ):
        product_query = _build_contextual_product_query(
            message,
            recent_product_topic.get("data", {}).get("query"),
        )
        product_result = search_products(product_query)
        set_pending_offer("product_filters")
        set_last_product_query(_extract_product_focus_query(product_query))
        brands, categories = _extract_topic_brands_and_categories(product_result)
        _remember_product_topic(_extract_product_focus_query(product_query), brands=brands, categories=categories)
        return _validate_or_return(
            "product_general",
            trace_start_index,
            _format_product_response(product_result),
        )

    if (
        recent_order_topic
        and not _user_does_not_remember_order_id(message)
        and _extract_order_id(message) is None
        and any(term in normalized for term in ["pedido", "orden", "estado", "total", "direccion", "metodo de pago", "pague"])
    ):
        order_id = recent_order_topic.get("data", {}).get("order_id")
        if order_id is not None:
            set_last_order_id(order_id)
            return _validate_or_return(
                "order_status_history",
                trace_start_index,
                _format_order_summary_options(get_order_status(order_id), get_shipment_details(order_id)),
            )

    return None


class RetailChallengeAgent:
    """
    Fallback legacy: mantiene reglas duras y continuidad básica,
    pero el agente principal debe ser el híbrido con LLM.
    """

    def __init__(self, streaming: bool = False):
        self.streaming = streaming

    def __call__(self, message: str):
        trace_start_index = get_tool_trace_length()
        pending_intent = get_pending_intent()
        pending_offer = get_pending_offer()
        session_customer = get_session_customer()
        skip_tool_validation = False
        message = (message or "").strip()

        if detect_prompt_injection(message):
            return (
                "No puedo seguir instrucciones que intenten omitir las reglas de seguridad "
                "o exponer datos no autorizados."
            )

        if session_customer is not None and _get_memory_order_id() is not None and _is_card_secret_request(message):
            return (
                "No puedo compartir el número completo, el CVV ni otros datos sensibles de la tarjeta. "
                "Por seguridad solo puedo confirmar el último cuatro y el banco emisor."
            )

        identifier = _extract_identifier(message)
        if identifier and (session_customer is None or _is_auth_pending_state(pending_intent, pending_offer)):
            return _handle_identifier_auth_flow(identifier, trace_start_index)

        if session_customer is None and _is_auth_pending_state(pending_intent, pending_offer):
            if _looks_like_customer_identifier_attempt(message):
                return _format_auth_failure_response()
            return (
                "Para seguir con eso necesito validar tu identidad. "
                "Compárteme por favor tu cédula o el teléfono asociado a la cuenta."
            )

        last_order_id = _get_memory_order_id()

        if session_customer is not None and last_order_id is not None:
            explicit_order_id = _extract_order_id(message)
            target_order_id = explicit_order_id or last_order_id
            contextual_intent = _intent_from_contextual_order_followup(message, target_order_id)

            if contextual_intent is not None:
                if contextual_intent == "order_status_history" and _wants_full_order_history(message):
                    status_result = get_order_status(target_order_id)
                    history_result = get_order_history(target_order_id)
                    shipment_result = get_shipment_details(target_order_id)
                    return _validate_or_return(
                        "order_status_history",
                        trace_start_index,
                        _format_order_history_response(history_result, shipment_result, status_result)
                    )

                response = _resolve_selected_order_intent(contextual_intent, target_order_id)
                if contextual_intent == "order_status_history":
                    set_pending_offer("order_summary_options")
                return _validate_or_return(contextual_intent, trace_start_index, response)

        if pending_offer == "order_summary_options" and _is_affirmative_reply(message):
            return (
                "Claro. ¿Qué prefieres revisar del pedido: el total pagado, el método de pago, "
                "la dirección de entrega o el historial completo?"
            )

        if pending_offer == "order_summary_options":
            last_order_id = _get_memory_order_id()

            if _wants_full_order_history(message) and last_order_id:
                status_result = get_order_status(last_order_id)
                history_result = get_order_history(last_order_id)
                shipment_result = get_shipment_details(last_order_id)
                _clear_pending_flow()
                return _validate_or_return(
                    "order_status_history",
                    trace_start_index,
                    _format_order_history_response(history_result, shipment_result, status_result)
                )

            followup_intent = _intent_from_order_summary_followup(message)
            if followup_intent is not None and last_order_id is not None:
                response = _resolve_selected_order_intent(followup_intent, last_order_id)
                return _validate_or_return(followup_intent, trace_start_index, response)

        if _is_context_recall_message(message):
            last_product_query = _get_memory_product_query()
            if last_product_query:
                return (
                    f"Sí, veníamos hablando de {_display_product_context(last_product_query)}. "
                    "Si quieres, sigo con opciones, disponibilidad o promociones sobre eso."
                )

            topic_recall_response = _build_topic_recall_response()
            if topic_recall_response:
                return topic_recall_response

        memory_recall_response = _resolve_topic_recall_with_memory(message, trace_start_index)
        if memory_recall_response is not None:
            return memory_recall_response

        preclassified_intent_data = _classify_intent(
            message,
            session_customer=session_customer,
            pending_intent=pending_intent,
            pending_offer=pending_offer,
        )
        preclassified_intent = preclassified_intent_data["intent"]

        if pending_offer == "returns_policy_followup" and _is_returns_policy_followup(message):
            set_pending_offer("returns_policy_followup")
            return _format_returns_policy_followup_response(message)

        if preclassified_intent == "policy":
            rag_result = search_policy_sections(message)
            response = format_policy_response(rag_result)
            _remember_policy_topic(message)
            if _is_returns_policy_query(message):
                set_pending_offer("returns_policy_followup")
                set_pending_intent(None)
            else:
                _clear_pending_flow()
            return _validate_or_return("policy", trace_start_index, response)

        if pending_offer == "product_filters" and (
            _is_promotion_followup_query(message) or _is_general_promotion_query(message)
        ):
            promotion_query = _build_contextual_promotion_query(message, _get_memory_product_query())
            promotion_result = _filter_promotion_result_by_context(
                search_promotions(promotion_query),
                promotion_query,
                _get_memory_product_query(),
            )
            set_pending_offer("promotion_followup")
            _remember_promotion_topic(promotion_query, _get_memory_product_query())
            return _validate_or_return(
                "promotion_general",
                trace_start_index,
                _format_promotion_response(promotion_result)
            )

        if pending_offer == "product_filters" and _is_product_filter_followup(message):
            last_product_query = _get_memory_product_query()
            if last_product_query:
                product_query = _build_contextual_product_query(message, last_product_query)
                product_result = search_products(f"{product_query} disponibilidad")
                set_pending_offer("product_filters")
                set_last_product_query(_extract_product_focus_query(product_query))
                brands, categories = _extract_topic_brands_and_categories(product_result)
                _remember_product_topic(_extract_product_focus_query(product_query), brands=brands, categories=categories)
                return _validate_or_return(
                    "product_general",
                    trace_start_index,
                    _format_product_response(product_result)
                )

        if pending_offer == "product_filters" and _is_product_followup_query(message):
            last_product_query = _get_memory_product_query()
            product_query = _build_contextual_product_query(message, last_product_query)
            product_result = search_products(product_query)
            set_pending_offer("product_filters")
            set_last_product_query(_extract_product_focus_query(product_query))
            brands, categories = _extract_topic_brands_and_categories(product_result)
            _remember_product_topic(_extract_product_focus_query(product_query), brands=brands, categories=categories)
            if _is_product_recommendation_query(message):
                followup_response = _format_product_recommendation_response(message, product_result)
            else:
                followup_response = _format_product_response(product_result)
            return _validate_or_return(
                "product_general",
                trace_start_index,
                followup_response
            )

        if pending_offer == "promotion_scope":
            if _is_general_confirmation(message):
                promotion_result = search_promotions("promociones generales")
                _clear_pending_flow()
                return _validate_or_return(
                    "promotion_general",
                    trace_start_index,
                    _format_promotion_response(promotion_result)
                )

            if _wants_catalog_categories(message):
                categories_result = list_product_categories()
                _clear_pending_flow()
                return _validate_or_return(
                    "catalog_categories",
                    trace_start_index,
                    _format_categories_response(categories_result)
                )

            if any(term in _normalize_text(message) for term in get_catalog_reference_terms()):
                promotion_query = _build_contextual_promotion_query(message, _get_memory_product_query())
                promotion_result = _filter_promotion_result_by_context(
                    search_promotions(promotion_query),
                    promotion_query,
                    _get_memory_product_query(),
                )
                _remember_promotion_topic(promotion_query, _get_memory_product_query())
                _clear_pending_flow()
                return _validate_or_return(
                    "promotion_general",
                    trace_start_index,
                    _format_promotion_response(promotion_result)
                )

        if pending_offer == "promotion_followup":
            if _is_product_followup_query(message):
                product_query = _build_contextual_product_query(message, _get_memory_product_query())
                product_result = search_products(product_query)
                set_pending_offer("product_filters")
                set_last_product_query(_extract_product_focus_query(product_query))
                brands, categories = _extract_topic_brands_and_categories(product_result)
                _remember_product_topic(_extract_product_focus_query(product_query), brands=brands, categories=categories)
                return _validate_or_return(
                    "product_general",
                    trace_start_index,
                    _format_product_response(product_result)
                )

            if _is_general_confirmation(message):
                promotion_result = search_promotions("promociones generales")
                set_pending_offer("promotion_followup")
                return _validate_or_return(
                    "promotion_general",
                    trace_start_index,
                    _format_promotion_response(promotion_result)
                )

            if _is_promotion_followup_query(message):
                promotion_query = _build_contextual_promotion_query(message, _get_memory_product_query())
                promotion_result = _filter_promotion_result_by_context(
                    search_promotions(promotion_query),
                    promotion_query,
                    _get_memory_product_query(),
                )
                set_pending_offer("promotion_followup")
                _remember_promotion_topic(promotion_query, _get_memory_product_query())
                return _validate_or_return(
                    "promotion_general",
                    trace_start_index,
                    _format_promotion_response(promotion_result)
                )

        if _wants_last_order(message) or _wants_order_by_status(message):
            inferred_intent = pending_intent
            if inferred_intent is None:
                address_intent = _resolve_address_intent(
                    message,
                    last_order_id=_get_memory_order_id() if session_customer is not None else None
                )
                if address_intent == "order_delivery_address":
                    inferred_intent = "order_delivery_address"
                else:
                    inferred_intent = _classify_intent(
                        message,
                        session_customer=session_customer,
                        pending_intent=pending_intent,
                        pending_offer=pending_offer,
                    )["intent"]
            inferred_intent = _normalize_recent_order_intent(inferred_intent)

            if session_customer is None:
                set_pending_intent(inferred_intent)
                set_pending_offer("resolve_recent_order_after_auth")
                return _auth_prompt("lo de tu pedido")

            selection_result = get_customer_orders_for_selection(limit=10)
            selected_order_id = _select_order_from_summary(selection_result, message)

            if selected_order_id is None:
                return _validate_or_return(
                    inferred_intent,
                    trace_start_index,
                    _format_customer_orders_summary(selection_result)
                )

            response = _resolve_selected_order_intent(inferred_intent, selected_order_id)
            if inferred_intent == "order_status_history":
                set_pending_offer("order_summary_options")
            return _validate_or_return(inferred_intent, trace_start_index, response)

        if _user_does_not_remember_order_id(message):
            if session_customer is None:
                set_pending_intent(_normalize_recent_order_intent(pending_intent))
                set_pending_offer("show_recent_orders_after_auth")
                return _auth_prompt("la búsqueda de tus pedidos")

            summary_result = (
                get_customer_orders_for_selection(limit=10)
                if pending_intent in ["order_amount", "order_items", "order_payment_method", "order_delivery_address"]
                else get_customer_orders_summary(limit=5)
            )
            return _validate_or_return(
                pending_intent if pending_intent in ["order_amount", "order_items", "order_payment_method", "order_delivery_address"] else "order_status_history",
                trace_start_index,
                _format_customer_orders_summary(summary_result)
            )

        if session_customer is not None:
            explicit_order_id = _extract_order_id(message)
            address_intent = _resolve_address_intent(message, last_order_id=last_order_id)
            target_order_id = explicit_order_id or last_order_id

            if address_intent == "order_delivery_address" and target_order_id is not None:
                set_last_order_id(target_order_id)
                address_result = get_order_delivery_address(target_order_id)
                return _validate_or_return(
                    "order_delivery_address",
                    trace_start_index,
                    _format_order_delivery_address_response(address_result)
                )

            if address_intent == "customer_address":
                address_result = get_customer_default_address()
                return _validate_or_return(
                    "customer_address",
                    trace_start_index,
                    _format_customer_address_response(address_result)
                )

        if session_customer is None:
            address_intent = _resolve_address_intent(message)
            if address_intent is not None:
                explicit_order_id = _extract_order_id(message)
                if explicit_order_id is not None:
                    set_last_order_id(explicit_order_id)
                pending_address_intent = (
                    "order_delivery_address"
                    if explicit_order_id is not None or _wants_last_order(message)
                    else "customer_address"
                )
                set_pending_intent(pending_address_intent)
                return _auth_prompt("esa dirección")

        explicit_order_id = _extract_order_id(message)
        if explicit_order_id and session_customer is not None and pending_intent in [
            "order_amount", "order_items", "order_payment_method", "order_status_history", "returns_order_case", "order_delivery_address"
        ]:
            response = _resolve_selected_order_intent(pending_intent, explicit_order_id)
            if pending_intent == "order_status_history":
                set_pending_offer("order_summary_options")
            return _validate_or_return(pending_intent, trace_start_index, response)

        intent_data = preclassified_intent_data
        intent = intent_data["intent"]

        auth_check = require_auth_if_needed(intent_data)
        if not auth_check["allowed"]:
            explicit_order_id = _extract_order_id(message)
            if explicit_order_id is not None:
                set_last_order_id(explicit_order_id)
                _remember_order_topic(explicit_order_id, intent)
            set_pending_intent(intent)
            return _auth_prompt("esa consulta")

        if intent == "greeting":
            response = _greeting_response()

        elif intent == "faq":
            response = _faq_response(message)
            _clear_pending_flow()

        elif intent == "policy":
            rag_result = search_policy_sections(message)
            response = format_policy_response(rag_result)
            _remember_policy_topic(message)
            if _is_returns_policy_query(message):
                set_pending_offer("returns_policy_followup")
                set_pending_intent(None)
            else:
                _clear_pending_flow()

        elif intent == "product_warranty":
            warranty_result = get_product_warranty_info(message)
            response = _format_warranty_response(warranty_result)
            _clear_pending_flow()

        elif intent == "product_general":
            product_query = _build_contextual_product_query(message, _get_memory_product_query())
            product_result = search_products(product_query)
            brands, categories = _extract_topic_brands_and_categories(product_result)
            if _is_product_recommendation_query(message):
                response = _format_product_recommendation_response(message, product_result)
            elif _is_purchase_guidance_query(message):
                response = _format_purchase_guidance_response(product_result)
            else:
                response = _format_product_response(product_result)
            set_pending_offer("product_filters")
            set_last_product_query(_extract_product_focus_query(product_query))
            _remember_product_topic(_extract_product_focus_query(product_query), brands=brands, categories=categories)

        elif intent == "promotion_general":
            if _is_general_promotion_query(message):
                response = _promotion_clarification_response()
                set_pending_offer("promotion_scope")
                skip_tool_validation = True
            else:
                promotion_query = _build_contextual_promotion_query(message, _get_memory_product_query())
                promotion_result = _filter_promotion_result_by_context(
                    search_promotions(promotion_query),
                    promotion_query,
                    _get_memory_product_query(),
                )
                response = _format_promotion_response(promotion_result)
                set_pending_offer("promotion_followup")
                _remember_promotion_topic(promotion_query, _get_memory_product_query())

        elif intent == "catalog_categories":
            categories_result = list_product_categories()
            response = _format_categories_response(categories_result)
            _clear_pending_flow()

        elif intent == "customer_address":
            address_result = get_customer_default_address()
            response = _format_customer_address_response(address_result)
            _clear_pending_flow()

        elif intent == "order_items":
            order_id = _extract_order_id(message) or _get_memory_order_id()
            if order_id is None:
                set_pending_intent(intent)
                return "Puedo decirte qué pediste. Necesito que me indiques el número del pedido."
            response = _resolve_selected_order_intent(intent, order_id)

        elif intent == "order_payment_method":
            order_id = _extract_order_id(message) or _get_memory_order_id()
            if order_id is None:
                selection_result = get_customer_orders_for_selection(limit=10)
                selected_order_id = _select_order_from_summary(selection_result, message)
                if selected_order_id is None:
                    set_pending_intent(intent)
                    return _validate_or_return(
                        "order_payment_method",
                        trace_start_index,
                        _format_customer_orders_summary(selection_result)
                    )
                order_id = selected_order_id
            response = _resolve_selected_order_intent(intent, order_id)

        elif intent == "order_amount":
            order_id = _extract_order_id(message) or _get_memory_order_id()
            if order_id is None:
                set_pending_intent(intent)
                return "Claro. Si me compartes el número del pedido, te digo ese dato enseguida."
            response = _resolve_selected_order_intent(intent, order_id)

        elif intent in ["order_status_history", "returns_order_case"]:
            order_id = _extract_order_id(message) or _get_memory_order_id()
            if order_id is None:
                set_pending_intent(intent)
                return "Claro. Si me compartes el número del pedido, reviso esa información por ti."

            set_last_order_id(order_id)
            status_result = get_order_status(order_id)
            history_result = get_order_history(order_id)
            shipment_result = get_shipment_details(order_id)
            _remember_order_topic(order_id, intent)

            if intent == "returns_order_case":
                response = _format_returns_case_response(history_result, shipment_result, status_result)
                _clear_pending_flow()

            elif status_result["success"] and not history_result["success"] and not shipment_result["success"]:
                response = _format_order_status_response(status_result)
                _clear_pending_flow()

            elif _wants_full_order_history(message):
                response = _format_order_history_response(history_result, shipment_result, status_result)
                _clear_pending_flow()
            else:
                response = _format_order_summary_options(status_result, shipment_result)
                set_pending_offer("order_summary_options")
                set_pending_intent(None)

        else:
            response = (
                "No estoy completamente seguro de lo que necesitas, pero puedo ayudarte con "
                "pedidos, productos o políticas. Cuéntame un poco más y te ayudo."
            )
            _clear_pending_flow()

        if not skip_tool_validation:
            tool_validation = validate_tool_usage(intent_data, trace_start_index)
            if not tool_validation["valid"]:
                return tool_validation["message"]

        return _finalize_response(message, intent, response)

    def reset_memory(self):
        reset_session()


class HybridRetailAgent:
    """
    Agente principal.
    LLM-first con reglas duras mínimas:
    - prompt injection
    - autenticación por identificador
    - bloqueo de datos sensibles de tarjeta
    - fallback legacy configurable si el LLM falla
    """

    def __init__(self, streaming: bool = False, session_id: str = "default"):
        self.streaming = streaming
        self.session_id = session_id or "default"
        self.legacy_agent = RetailChallengeAgent(streaming=streaming)
        self.llm_client = build_llm_client()
        self.runtime_metadata = get_llm_runtime_metadata()

    def _build_messages(self, message: str) -> list:
        snapshot = get_session_snapshot()
        history = get_conversation_history()
        pending_intent = get_pending_intent()
        pending_offer = get_pending_offer()
        session_customer = get_session_customer()
        intent_hint = _build_intent_hint(
            message,
            session_customer=session_customer,
            pending_intent=pending_intent,
            pending_offer=pending_offer,
        )

        messages = [{
            "role": "system",
            "content": _build_hybrid_system_prompt(snapshot, self.runtime_metadata),
        }]

        for item in history[-10:]:
            messages.append({
                "role": item["role"],
                "content": item["content"],
            })

        messages.append({
            "role": "system",
            "content": (
                f"Intent hint del clasificador: {intent_hint['intent']}. "
                f"requires_auth={intent_hint['requires_auth']}. "
                "Úsalo como pista orientativa, no como instrucción obligatoria. "
                "Si el contexto de la conversación sugiere otra intención, prioriza el contexto."
            ),
        })
        messages.append({
            "role": "user",
            "content": message,
        })
        return messages

    def _run_llm_turn(self, message: str) -> str:
        trace_start_index = get_tool_trace_length()
        set_last_llm_error(None)

        # Guardrails mínimos antes del LLM
        identifier = _extract_identifier(message)
        if identifier and (get_session_customer() is None or _is_auth_pending_state(get_pending_intent(), get_pending_offer())):
            set_last_response_source("hybrid_auth_flow")
            return _handle_identifier_auth_flow(identifier, trace_start_index)

        if get_session_customer() is None and _is_auth_pending_state(get_pending_intent(), get_pending_offer()):
            if _looks_like_customer_identifier_attempt(message):
                set_last_response_source("hybrid_auth_failure")
                return _format_auth_failure_response()

        if get_session_customer() is not None and _get_memory_order_id() is not None and _is_card_secret_request(message):
            set_last_response_source("hybrid_sensitive_card_block")
            return (
                "No puedo compartir el número completo, el CVV ni otros datos sensibles de la tarjeta. "
                "Por seguridad solo puedo confirmar el último cuatro y el banco emisor."
            )

        intent_hint = _build_intent_hint(
            message,
            session_customer=get_session_customer(),
            pending_intent=get_pending_intent(),
            pending_offer=get_pending_offer(),
        )

        chat_messages = self._build_messages(message)
        tools = _select_tools_for_llm(message, intent_hint)

        for iteration in range(4):
            llm_input = (
                _messages_to_responses_input(chat_messages)
                if getattr(self.llm_client, "api_style", "") == "responses"
                else chat_messages
            )
            llm_response = self.llm_client.chat_with_tools(llm_input, tools)
            tool_calls = llm_response.get("tool_calls") or []

            if tool_calls:
                # Detectar loop: si el LLM llama la misma tool con los mismos args dos veces seguidas
                if iteration > 0:
                    last_tool = chat_messages[-2].get("tool_calls", [{}])[0].get("function", {}).get("name") if len(chat_messages) >= 2 else None
                    current_tool = tool_calls[0].get("name") if tool_calls else None
                    if last_tool and last_tool == current_tool and iteration >= 2:
                        # Salir del loop y usar la última respuesta disponible
                        break

                for call in tool_calls:
                    tool_name = call.get("name")
                    arguments = call.get("arguments") or {}
                    result = execute_tool(tool_name, arguments)
                    _update_session_state_from_tool_result(tool_name, arguments, result)

                    chat_messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": call.get("id"),
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }],
                    })
                    chat_messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "content": _tool_result_to_content(tool_name, result),
                    })

                    if result.get("message") == "AUTH_REQUIRED":
                        set_pending_intent(intent_hint.get("intent"))
                        set_last_response_source("hybrid_auth_prompt")
                        return _auth_prompt("esa consulta")
                continue

            final_text = (llm_response.get("content") or "").strip()
            _update_session_state_from_final_response(intent_hint, trace_start_index, final_text)
            validation = validate_hybrid_response(intent_hint, trace_start_index, final_text)
            if not validation["valid"]:
                if iteration < 3:
                    chat_messages.append({
                        "role": "system",
                        "content": (
                            f"Tu respuesta anterior no fue válida: {validation['message']} "
                            "Inténtalo de nuevo. Usa una tool solo si necesitas datos verificables; "
                            "si no hace falta, responde con una aclaración breve o pide el dato faltante "
                            "sin inventar información."
                        ),
                    })
                    continue
                return validation["message"]

            set_last_response_source("llm")
            return final_text

        return (
            "No pude completar la consulta con suficiente información. "
            "¿Puedes darme un poco más de detalle sobre lo que necesitas?"
        )

    def __call__(self, message: str):
        set_active_session(self.session_id)
        set_last_response_source(None)
        set_last_llm_error(None)

        if detect_prompt_injection(message):
            set_last_response_source("guard")
            response = (
                "No puedo seguir instrucciones que intenten omitir las reglas de seguridad "
                "o exponer datos no autorizados."
            )
            add_conversation_turn("user", message)
            add_conversation_turn("assistant", response)
            return response

        if not self.llm_client.is_enabled():
            unavailable_reason = None
            if hasattr(self.llm_client, "get_unavailable_reason"):
                unavailable_reason = self.llm_client.get_unavailable_reason()
            if unavailable_reason:
                set_last_llm_error(unavailable_reason)

            if _llm_strict_mode_enabled():
                set_last_response_source("llm_unavailable")
                response = _build_llm_unavailable_response(self.llm_client, self.runtime_metadata)
                add_conversation_turn("user", message)
                response = _with_runtime_marker(response)
                add_conversation_turn("assistant", response)
                return response

            set_last_response_source("legacy_fallback")
            response = self.legacy_agent(message)
            add_conversation_turn("user", message)
            response = _with_runtime_marker(response)
            add_conversation_turn("assistant", response)
            return response

        try:
            # LLM primero siempre
            response = self._run_llm_turn(message)
        except LLMClientError as exc:
            set_last_llm_error(str(exc))
            if _llm_strict_mode_enabled():
                set_last_response_source("llm_error")
                response = (
                    "El modelo LLM no respondió correctamente y el modo estricto está activo. "
                    "No usaré fallback legacy en esta sesión."
                )
            else:
                set_last_response_source("legacy_fallback")
                response = self.legacy_agent(message)

        add_conversation_turn("user", message)
        response = _with_runtime_marker(response)
        add_conversation_turn("assistant", response)
        return response

    def reset_memory(self):
        set_active_session(self.session_id)
        reset_session()


class SafeFallbackAgent:
    def __init__(self, streaming: bool = False, boot_error: Optional[Exception] = None):
        self.streaming = streaming
        self.boot_error = boot_error

    def __call__(self, message: str):
        return (
            "Lo siento, en este momento no pude inicializar correctamente el agente. "
            "Intenta nuevamente en unos segundos."
        )

    def reset_memory(self):
        reset_session()


def create_agent_for_session(session_id: str, streaming: bool = False):
    try:
        if _AGENT_BOOT_ERROR is not None:
            return SafeFallbackAgent(streaming=streaming, boot_error=_AGENT_BOOT_ERROR)
        return HybridRetailAgent(streaming=streaming, session_id=session_id)
    except Exception as exc:
        return SafeFallbackAgent(streaming=streaming, boot_error=exc)


def create_agent(streaming: bool = False):
    default_session_id = os.getenv("AGENT_SESSION_ID", "default").strip() or "default"
    return create_agent_for_session(session_id=default_session_id, streaming=streaming)


def _run_cli():
    agent = create_agent()

    while True:
        try:
            user_input = input("Usuario: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in ["salir", "exit", "quit"]:
            break

        response = agent(user_input)
        print("Agente:", response)
        print("-" * 80)


if __name__ == "__main__":
    _run_cli()
