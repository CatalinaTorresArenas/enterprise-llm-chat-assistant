import re
from typing import Any, Dict, List

from core.guards import validate_tool_usage, normalize_text
from core.session_context import get_tool_trace_since


def _is_allowed_conversational_response_without_tools(response_text: str) -> bool:
    normalized = normalize_text(response_text)

    allowed_markers = [
        # autenticacion
        "comparteme tu cedula",
        "comparteme tu cédula",
        "comparteme tu telefono",
        "comparteme tu teléfono",
        "comparteme por favor tu cedula",
        "comparteme por favor tu cédula",
        "comparteme por favor tu telefono",
        "comparteme por favor tu teléfono",
        "necesito validar tu identidad",
        "necesito confirmar que la cuenta si es tuya",
        "necesito confirmar que la cuenta sí es tuya",
        "primero necesito validar tu identidad",
        "primero necesito confirmar tu identidad",
        "antes necesito confirmar que la cuenta si es tuya",
        "antes necesito confirmar que la cuenta sí es tuya",

        # solicitud de dato faltante
        "necesito que me indiques el numero del pedido",
        "necesito que me indiques el número del pedido",
        "si me compartes el numero del pedido",
        "si me compartes el número del pedido",
        "si me dices el numero del pedido",
        "si me dices el número del pedido",
        "puedo mostrarte tus pedidos recientes",
        "tambien puedo mostrarte tus pedidos recientes",
        "también puedo mostrarte tus pedidos recientes",
        "dime cual quieres revisar",
        "dime cuál quieres revisar",

        # aclaraciones conversacionales validas
        "que prefieres revisar",
        "qué prefieres revisar",
        "te puedo mostrar",
        "puedo ayudarte con",
        "si quieres te muestro",
        "si quieres tambien te puedo decir",
        "si quieres también te puedo decir",
        "si quieres sigo con",
        "cuentame que quieres revisar",
        "cuéntame qué quieres revisar",
        "no estoy completamente seguro de lo que necesitas",
        "no pude validar la cuenta con ese dato",
        "no pude confirmar la cuenta con ese dato",
    ]

    return any(marker in normalized for marker in allowed_markers)


def _is_allowed_intent_guidance_without_tools(intent_data: Dict[str, Any], response_text: str) -> bool:
    intent = intent_data.get("intent")
    normalized = normalize_text(response_text)

    guidance_intents = {
        "greeting",
        "faq",
        "product_general",
        "promotion_general",
        "product_warranty",
        "policy",
    }
    if intent not in guidance_intents:
        return False

    generic_markers = [
        "si quieres",
        "si me dices",
        "te puedo ayudar",
        "te puedo orientar",
        "te puedo mostrar",
        "puedo ayudarte",
        "puedo orientarte",
        "para ayudarte mejor",
        "para afinar",
        "dime si",
        "que prefieres",
        "qué prefieres",
        "que buscas",
        "qué buscas",
        "que necesitas",
        "qué necesitas",
    ]

    intent_specific_markers = {
        "product_general": [
            "marca",
            "presupuesto",
            "uso le vas a dar",
            "tipo de producto",
            "te conviene",
            "algo economico",
            "algo económico",
        ],
        "promotion_general": [
            "promociones generales",
            "categoria o producto",
            "categoría o producto",
            "de que categoria",
            "de qué categoría",
            "de que producto",
            "de qué producto",
        ],
        "product_warranty": [
            "producto exacto",
            "categoria exacta",
            "categoría exacta",
        ],
        "policy": [
            "si es por devolucion",
            "si es por devolución",
            "si es por garantia",
            "si es por garantía",
            "si es por envio",
            "si es por envío",
        ],
    }

    return any(marker in normalized for marker in generic_markers + intent_specific_markers.get(intent, []))


def _contains_structured_claims_without_evidence(response_text: str) -> bool:
    normalized = normalize_text(response_text)

    money_pattern = re.search(r"\$\s*\d", response_text or "")
    date_pattern = re.search(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", response_text or "")
    order_fact_markers = [
        "pedido",
        "orden",
        "subtotal",
        "iva",
        "total",
        "transportadora",
        "guia",
        "guía",
        "tracking",
        "entregado",
        "enviado",
        "cancelado",
        "direccion de entrega",
        "dirección de entrega",
        "ultimos 4",
        "últimos 4",
    ]

    has_order_marker = any(marker in normalized for marker in order_fact_markers)
    return bool(money_pattern or date_pattern or has_order_marker)


def validate_hybrid_response(
    intent_data: Dict[str, Any],
    trace_start_index: int,
    response_text: str,
) -> Dict[str, Any]:
    response_text = response_text or ""
    normalized = normalize_text(response_text)
    traces = get_tool_trace_since(trace_start_index)

    if not response_text.strip():
        return {
            "valid": False,
            "message": "El agente no pudo construir una respuesta final valida.",
        }

 
    if not traces and _is_allowed_conversational_response_without_tools(response_text):
        return {"valid": True}

    if not traces and _is_allowed_intent_guidance_without_tools(intent_data, response_text):
        return {"valid": True}

    if not traces and _contains_structured_claims_without_evidence(response_text):
        return {
            "valid": False,
            "message": "La respuesta parece incluir datos concretos sin evidencia de herramientas en este turno.",
        }


    tool_check = validate_tool_usage(intent_data, trace_start_index)
    if not tool_check["valid"]:
        return tool_check


    if traces and not normalized:
        return {
            "valid": False,
            "message": "Hubo consulta de datos, pero no se pudo construir una respuesta final valida.",
        }

    return {"valid": True}


def build_tool_context(trace_start_index: int) -> List[Dict[str, Any]]:
    return get_tool_trace_since(trace_start_index)
