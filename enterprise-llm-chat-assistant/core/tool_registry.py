from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from core.policy_rag import search_policy_sections
from core.session_context import get_session_customer
from core.tools import (
    authenticate_customer,
    get_customer_default_address,
    get_customer_orders_for_selection,
    get_customer_orders_summary,
    get_order_amounts,
    get_order_items,
    get_order_delivery_address,
    get_order_history,
    get_order_payment_method,
    get_order_status,
    get_product_warranty_info,
    get_shipment_details,
    list_product_categories,
    search_products,
    search_promotions,
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., Dict[str, Any]]
    requires_auth: bool = False
    sensitive: bool = False


def _order_id_schema() -> Dict[str, Any]:
    return {
        "type": "string",
        "description": "Identificador del pedido. Puede ser numerico o alfanumerico, por ejemplo 10234 o GR501146.",
    }


def _limit_schema(default_max: int = 10) -> Dict[str, Any]:
    return {
        "type": "integer",
        "minimum": 1,
        "maximum": default_max,
        "description": f"Cantidad maxima de resultados a devolver. Valor entre 1 y {default_max}.",
    }


def _build_registry() -> Dict[str, ToolSpec]:
    return {
        "authenticate_customer": ToolSpec(
            name="authenticate_customer",
            description=(
                "Valida la identidad del cliente usando una cedula o numero de telefono. "
                "Usa esta herramienta cuando el usuario comparta su identificacion o cuando falte autenticacion."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": "Cedula o telefono del cliente tal como lo compartio.",
                    },
                },
                "required": ["identifier"],
                "additionalProperties": False,
            },
            handler=lambda identifier: authenticate_customer(identifier),
        ),
        "search_policy_sections": ToolSpec(
            name="search_policy_sections",
            description=(
                "Recupera secciones relevantes de las politicas markdown para responder con base documental. "
                "Usa esta herramienta para garantias, devoluciones, envios, reembolsos o politicas generales."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Pregunta del usuario sobre politicas, devoluciones, garantias o envios.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=lambda query: search_policy_sections(query),
        ),
        "search_products": ToolSpec(
            name="search_products",
            description=(
                "Busca productos del catalogo, precios, disponibilidad y atributos comerciales. "
                "Usa esta herramienta para consultas sobre categorias, marcas, productos, stock o compra."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Consulta de catalogo, disponibilidad, marca, categoria o producto.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=lambda query: search_products(query),
        ),
        "search_promotions": ToolSpec(
            name="search_promotions",
            description=(
                "Busca promociones activas por categoria, producto, marca o de forma general. "
                "Usa esta herramienta cuando el usuario pregunte por descuentos, ofertas o rebajas."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Consulta de promociones.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=lambda query: search_promotions(query),
        ),
        "list_product_categories": ToolSpec(
            name="list_product_categories",
            description="Lista las categorias disponibles del catalogo.",
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            handler=lambda: list_product_categories(),
        ),
        "get_product_warranty_info": ToolSpec(
            name="get_product_warranty_info",
            description=(
                "Obtiene garantia y plazo de devolucion de productos o categorias del catalogo. "
                "Usa esta herramienta cuando la consulta de garantia este enfocada en productos concretos o categorias."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Nombre del producto o categoria.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=lambda query: get_product_warranty_info(query),
        ),
        "get_customer_orders_summary": ToolSpec(
            name="get_customer_orders_summary",
            description=(
                "Lista pedidos recientes del cliente autenticado. "
                "Usa esta herramienta cuando el usuario no recuerde el numero del pedido o quiera ver sus pedidos recientes."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "limit": _limit_schema(10),
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=lambda limit=5: get_customer_orders_summary(limit=limit),
            requires_auth=True,
            sensitive=True,
        ),
        "get_customer_orders_for_selection": ToolSpec(
            name="get_customer_orders_for_selection",
            description=(
                "Lista pedidos recientes con datos utiles para seleccionar uno antes de profundizar. "
                "Usa esta herramienta cuando necesites que el usuario elija uno de sus pedidos."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "limit": _limit_schema(10),
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=lambda limit=10: get_customer_orders_for_selection(limit=limit),
            requires_auth=True,
            sensitive=True,
        ),
        "get_order_status": ToolSpec(
            name="get_order_status",
            description=(
                "Consulta el estado del pedido autenticado. "
                "Usa esta herramienta para estado actual, fecha del pedido, despacho, entrega o metodo de entrega."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": _order_id_schema(),
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
            handler=lambda order_id: get_order_status(order_id),
            requires_auth=True,
            sensitive=True,
        ),
        "get_order_amounts": ToolSpec(
            name="get_order_amounts",
            description=(
                "Consulta subtotal, impuestos, envio y total de un pedido autenticado."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": _order_id_schema(),
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
            handler=lambda order_id: get_order_amounts(order_id),
            requires_auth=True,
            sensitive=True,
        ),
        "get_order_items": ToolSpec(
            name="get_order_items",
            description=(
                "Consulta los productos, cantidades y estados incluidos en un pedido autenticado."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": _order_id_schema(),
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
            handler=lambda order_id: get_order_items(order_id),
            requires_auth=True,
            sensitive=True,
        ),
        "get_order_payment_method": ToolSpec(
            name="get_order_payment_method",
            description=(
                "Consulta el medio de pago y detalles asociados de un pedido autenticado. "
                "Solo devuelve informacion segura como banco, tipo de tarjeta y ultimos cuatro digitos."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": _order_id_schema(),
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
            handler=lambda order_id: get_order_payment_method(order_id),
            requires_auth=True,
            sensitive=True,
        ),
        "get_customer_default_address": ToolSpec(
            name="get_customer_default_address",
            description="Consulta la direccion principal del cliente autenticado.",
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            handler=lambda: get_customer_default_address(),
            requires_auth=True,
            sensitive=True,
        ),
        "get_order_delivery_address": ToolSpec(
            name="get_order_delivery_address",
            description="Consulta la direccion de entrega asociada a un pedido autenticado.",
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": _order_id_schema(),
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
            handler=lambda order_id: get_order_delivery_address(order_id),
            requires_auth=True,
            sensitive=True,
        ),
        "get_order_history": ToolSpec(
            name="get_order_history",
            description=(
                "Consulta el historial de tracking del pedido autenticado."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": _order_id_schema(),
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
            handler=lambda order_id: get_order_history(order_id),
            requires_auth=True,
            sensitive=True,
        ),
        "get_shipment_details": ToolSpec(
            name="get_shipment_details",
            description=(
                "Consulta transportadora, guia y fechas del envio de un pedido autenticado."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": _order_id_schema(),
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
            handler=lambda order_id: get_shipment_details(order_id),
            requires_auth=True,
            sensitive=True,
        ),
    }


TOOL_REGISTRY = _build_registry()


def get_tool_specs() -> Dict[str, ToolSpec]:
    return dict(TOOL_REGISTRY)


def get_tool_spec(name: str) -> ToolSpec:
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown tool: {name}")
    return TOOL_REGISTRY[name]


def get_tool_definitions_for_llm() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
            "strict": True,
        }
        for spec in TOOL_REGISTRY.values()
    ]


def execute_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    try:
        spec = get_tool_spec(name)
    except KeyError:
        return {
            "success": False,
            "message": f"UNKNOWN_TOOL:{name}",
        }

    if spec.requires_auth and get_session_customer() is None:
        return {
            "success": False,
            "message": "AUTH_REQUIRED",
            "tool_name": name,
        }

    safe_arguments = dict(arguments or {})

    if "order_id" in safe_arguments and safe_arguments["order_id"] is not None:
        safe_arguments["order_id"] = str(safe_arguments["order_id"]).strip()

    try:
        return spec.handler(**safe_arguments)
    except TypeError as exc:
        return {
            "success": False,
            "message": f"INVALID_TOOL_ARGUMENTS:{name}",
            "detail": str(exc),
        }
    except Exception as exc:
        return {
            "success": False,
            "message": f"TOOL_EXECUTION_ERROR:{name}",
            "detail": str(exc),
        }