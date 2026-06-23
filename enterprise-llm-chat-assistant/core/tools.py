import csv
import os
import re
import unicodedata

from core.session_context import (
    add_tool_trace,
    set_session_customer,
    get_session_customer,
)

BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "csv")


def _read_csv(filename: str):
    path = os.path.join(BASE_PATH, filename)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


customers = _read_csv("customers.csv")
orders = _read_csv("orders.csv")
order_items = _read_csv("order_items.csv")
products = _read_csv("products.csv")
categories = _read_csv("categories.csv")
brands = _read_csv("brands.csv")
stock = _read_csv("stock.csv")
shipments = _read_csv("shipments.csv")
tracking = _read_csv("tracking.csv")
addresses = _read_csv("addresses.csv")
cards = _read_csv("cards.csv")
promotions = _read_csv("promotions.csv")

CATALOG_ALIASES = {
    "telefonos": ["telefono", "telefonos", "celular", "celulares", "smartphone", "movil", "iphone", "galaxy"],
    "celulares": ["telefono", "telefonos", "celular", "celulares", "smartphone", "movil", "iphone", "galaxy"],
    "laptops": ["laptop", "laptops", "portatil", "portatiles", "notebook", "macbook", "computador"],
    "portatiles": ["laptop", "laptops", "portatil", "portatiles", "notebook", "macbook", "computador"],
    "televisores": ["televisor", "televisores", "tv", "smart tv"],
    "tv": ["televisor", "televisores", "tv", "smart tv"],
    "tablets": ["tablet", "tablets", "tableta", "ipad"],
    "audifonos": ["audifono", "audifonos", "auriculares"],
    "monitores": ["monitor", "monitores", "pantalla"],
    "colchones": ["colchon", "colchones"],
}


def _normalize_catalog_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).lower().strip())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_generic_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).strip())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_order_id(value) -> str | None:
    if _is_missing(value):
        return None
    normalized = unicodedata.normalize("NFKD", str(value).upper().strip())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"[^A-Z0-9]+", "", normalized)
    return normalized or None


def _same_order_id(a, b) -> bool:
    na = _normalize_order_id(a)
    nb = _normalize_order_id(b)
    return na is not None and nb is not None and na == nb


def _safe_order_id(value):
    if _is_missing(value):
        return None
    text = str(value).strip()
    as_int = _as_int(text, None)
    if as_int is not None and str(as_int) == text:
        return as_int
    return text


def _is_missing(value) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "nat"}


def _as_str(value):
    return None if _is_missing(value) else str(value)


def _as_int(value, default=None):
    if _is_missing(value):
        return default
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return default


def _as_float(value, default=None):
    if _is_missing(value):
        return default
    try:
        return float(str(value).strip())
    except ValueError:
        return default


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if _is_missing(value):
        return False
    return str(value).strip().lower() == "true"


def _rows_where(rows, predicate):
    return [row for row in rows if predicate(row)]


def _first_or_none(rows):
    return rows[0] if rows else None


def _index_by(rows, key):
    indexed = {}
    for row in rows:
        indexed[row[key]] = row
    return indexed


def _sort_rows(rows, key, reverse=False):
    return sorted(rows, key=lambda row: str(row.get(key, "")), reverse=reverse)


def _head(rows, limit):
    return rows[:limit]


def get_catalog_reference_terms():
    stopwords = {
        "de", "del", "la", "el", "los", "las", "y", "con", "para", "en",
        "sin", "set", "smart", "pro", "max", "ultra", "queen", "king",
        "in", "4k", "5g", "plus"
    }
    terms = set()

    values = [row["name"] for row in categories] + [row["name"] for row in products]
    for value in values:
        normalized = _normalize_catalog_text(value)
        if not normalized:
            continue

        terms.add(normalized)

        for token in normalized.split():
            if len(token) >= 4 and token not in stopwords:
                terms.add(token)

    for alias_group, aliases in CATALOG_ALIASES.items():
        terms.add(alias_group)
        for alias in aliases:
            terms.add(_normalize_catalog_text(alias))

    return terms


def _query_tokens(query: str):
    normalized = _normalize_catalog_text(query)
    stopwords = {
        "que", "cual", "cuales", "quiero", "saber", "la", "el", "los", "las",
        "de", "del", "para", "tiene", "tienen", "garantia", "garantias",
        "producto", "productos", "sobre", "me", "podrias", "decir",
        "promocion", "promociones", "descuento", "descuentos", "oferta", "ofertas",
        "rebaja", "rebajas", "vigente", "vigentes", "tengo", "tienes", "tienen",
        "muéstrame", "muestrame", "hay", "quieres", "ver", "mostrar"
    }
    base_tokens = [token for token in normalized.split() if len(token) >= 3 and token not in stopwords]
    expanded_tokens = []

    for token in base_tokens:
        expanded_tokens.append(token)
        if token.endswith("es") and len(token) > 4:
            expanded_tokens.append(token[:-2])
        elif token.endswith("s") and len(token) > 4:
            expanded_tokens.append(token[:-1])

    normalized_query = " ".join(expanded_tokens)
    for alias_group, aliases in CATALOG_ALIASES.items():
        if alias_group in normalized_query:
            expanded_tokens.extend(aliases)

    return list(dict.fromkeys(expanded_tokens))


def _normalize_phone(phone: str) -> str:
    digits = "".join(ch for ch in str(phone) if ch.isdigit())
    if digits.startswith("57") and len(digits) > 10:
        digits = digits[2:]
    return digits


def _normalize_document_identifier(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value).upper().strip())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"[^A-Z0-9]+", "", normalized)
    return normalized


def _iter_identifier_candidates(message: str):
    raw_message = str(message).strip()
    if not raw_message:
        return []

    candidates = [raw_message]
    patterns = [
        r"(?:\+?\s*57[\s\-]*)?(?:3\d[\s\-]*){4}\d{2,4}",
        r"\b[A-Za-z]{1,4}[\s\-#]*\d{3,}\b",
        r"\b\d{5,}\b",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, raw_message, flags=re.IGNORECASE):
            candidates.append(match.group(0))

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        normalized_candidate = candidate.strip()
        if normalized_candidate and normalized_candidate not in seen:
            seen.add(normalized_candidate)
            unique_candidates.append(normalized_candidate)

    return unique_candidates


def extract_customer_identifier(message: str):
    raw_message = str(message).strip()
    if not raw_message:
        return None

    dni_lookup = {
        _normalize_document_identifier(row["dni"]): str(row["dni"]).strip().upper()
        for row in customers
        if _normalize_document_identifier(row["dni"])
    }
    phone_values = [str(row["phone"]).strip() for row in customers]
    normalized_phone_lookup = {
        _normalize_phone(phone): phone
        for phone in phone_values
        if _normalize_phone(phone)
    }

    for candidate in _iter_identifier_candidates(raw_message):
        normalized_document = _normalize_document_identifier(candidate)
        if normalized_document in dni_lookup:
            return dni_lookup[normalized_document]

        normalized_phone = _normalize_phone(candidate)
        if normalized_phone in normalized_phone_lookup:
            return normalized_phone

    return None


def authenticate_customer(identifier: str):
    clean_identifier = str(identifier).strip().upper()
    normalized_document = _normalize_document_identifier(identifier)
    normalized_input_phone = _normalize_phone(identifier)

    result = _first_or_none(
        _rows_where(
            customers,
            lambda row: (
                _normalize_document_identifier(row["dni"]) == normalized_document or
                str(row["dni"]).strip().upper() == clean_identifier or
                str(row["phone"]).strip() == str(identifier).strip() or
                _normalize_phone(row["phone"]) == normalized_input_phone
            ),
        )
    )

    if result is None:
        output = {"success": False, "message": "No encontré un cliente con ese documento o teléfono."}
        add_tool_trace("authenticate_customer", {"identifier": identifier}, output)
        return output

    output = {
        "success": True,
        "customer_id": _as_int(result["customer_id"]),
        "name": result["name"],
        "account_status": result["account_status"],
        "auth_method": "dni" if _normalize_document_identifier(result["dni"]) == normalized_document else "phone",
    }

    set_session_customer(output["customer_id"], output["name"])
    add_tool_trace("authenticate_customer", {"identifier": identifier}, output)
    return output


def _require_authenticated_customer():
    session_customer = get_session_customer()

    if session_customer is None:
        return {
            "success": False,
            "message": "AUTH_REQUIRED"
        }

    return {
        "success": True,
        "customer_id": session_customer["customer_id"],
        "display_name": session_customer["display_name"]
    }


def _get_customer_order(order_id):
    auth = _require_authenticated_customer()

    if not auth["success"]:
        return auth

    customer_id = auth["customer_id"]
    normalized_target_order_id = _normalize_order_id(order_id)

    result = _first_or_none(
        _rows_where(
            orders,
            lambda row: (
                _as_int(row["customer_id"]) == customer_id and
                _same_order_id(row["order_id"], normalized_target_order_id)
            ),
        )
    )

    if result is None:
        return {
            "success": False,
            "message": "No encontré un pedido con ese número asociado a tu cuenta."
        }

    return {
        "success": True,
        "customer_id": customer_id,
        "order_row": result
    }


def get_order_status(order_id):
    lookup = _get_customer_order(order_id)

    if not lookup["success"]:
        output = lookup
        add_tool_trace("get_order_status", {"order_id": order_id}, output)
        return output

    order = lookup["order_row"]
    output = {
        "success": True,
        "order_id": _safe_order_id(order["order_id"]),
        "customer_id": _as_int(order["customer_id"]),
        "status": str(order["status"]),
        "order_date": _as_str(order["order_date"]),
        "shipped_at": _as_str(order["shipped_at"]),
        "delivered_at": _as_str(order["delivered_at"]),
        "cancelled_at": _as_str(order["cancelled_at"]),
        "delivery_method": _as_str(order["delivery_method"]),
    }

    add_tool_trace("get_order_status", {"order_id": order_id}, output)
    return output


def get_order_amounts(order_id):
    lookup = _get_customer_order(order_id)

    if not lookup["success"]:
        output = lookup
        add_tool_trace("get_order_amounts", {"order_id": order_id}, output)
        return output

    order = lookup["order_row"]
    output = {
        "success": True,
        "order_id": _safe_order_id(order["order_id"]),
        "customer_id": _as_int(order["customer_id"]),
        "subtotal": _as_float(order["subtotal"], 0.0),
        "tax": _as_float(order["tax"], 0.0),
        "shipping_cost": _as_float(order["shipping_cost"], 0.0),
        "total_amount": _as_float(order["total_amount"], 0.0),
        "status": str(order["status"]),
    }

    add_tool_trace("get_order_amounts", {"order_id": order_id}, output)
    return output


def get_order_payment_method(order_id):
    lookup = _get_customer_order(order_id)

    if not lookup["success"]:
        output = lookup
        add_tool_trace("get_order_payment_method", {"order_id": order_id}, output)
        return output

    order = lookup["order_row"]
    payment_method = _as_str(order["payment_method"])
    card_id = _as_int(order["card_id"])

    output = {
        "success": True,
        "order_id": _safe_order_id(order["order_id"]),
        "payment_method": payment_method,
        "payment_confirmed_at": _as_str(order["payment_confirmed_at"]),
        "card": None,
    }

    if card_id is not None:
        card = _first_or_none(_rows_where(cards, lambda row: _as_int(row["card_id"]) == card_id))
        if card is not None:
            output["card"] = {
                "card_type": str(card["card_type"]),
                "bank": str(card["bank"]),
                "last_four": str(card["last_four"]),
            }

    add_tool_trace("get_order_payment_method", {"order_id": order_id}, output)
    return output


def get_order_items(order_id):
    lookup = _get_customer_order(order_id)

    if not lookup["success"]:
        output = lookup
        add_tool_trace("get_order_items", {"order_id": order_id}, output)
        return output

    matching_items = _rows_where(order_items, lambda row: _same_order_id(row["order_id"], order_id))
    if not matching_items:
        output = {
            "success": False,
            "message": "No encontré productos asociados a ese pedido."
        }
        add_tool_trace("get_order_items", {"order_id": order_id}, output)
        return output

    products_by_id = _index_by(products, "product_id")
    canonical_order_id = _safe_order_id(lookup["order_row"]["order_id"])
    items = []

    for row in matching_items:
        product = products_by_id.get(row["product_id"], {})
        qty = _as_int(row["qty"], 0) or 0
        unit_price = _as_float(row["unit_price"], 0.0) or 0.0
        items.append({
            "item_id": _as_int(row["item_id"]),
            "order_id": canonical_order_id,
            "product_id": _as_int(row["product_id"]),
            "product_name": _as_str(product.get("name")) or "Producto no encontrado",
            "qty": qty,
            "unit_price": unit_price,
            "line_total": qty * unit_price,
            "item_status": _as_str(row["item_status"]) or "unknown",
            "warranty_expires_at": _as_str(row["warranty_expires_at"]),
            "return_deadline": _as_str(row["return_deadline"]),
        })

    output = {
        "success": True,
        "order_id": canonical_order_id,
        "items": items,
    }
    add_tool_trace("get_order_items", {"order_id": order_id}, output)
    return output


def get_customer_default_address():
    auth = _require_authenticated_customer()

    if not auth["success"]:
        output = auth
        add_tool_trace("get_customer_default_address", {}, output)
        return output

    customer_id = auth["customer_id"]
    customer_addresses = _rows_where(addresses, lambda row: _as_int(row["customer_id"]) == customer_id)

    if not customer_addresses:
        output = {
            "success": False,
            "message": "No se encontró una dirección asociada al cliente autenticado."
        }
        add_tool_trace("get_customer_default_address", {}, output)
        return output

    default_rows = _rows_where(customer_addresses, lambda row: _as_bool(row["is_default"]))
    if default_rows:
        customer_addresses = default_rows

    address = customer_addresses[0]
    output = {
        "success": True,
        "address_id": _as_int(address["address_id"]),
        "address_line1": str(address["address_line1"]),
        "address_line2": _as_str(address["address_line2"]),
        "city": str(address["city"]),
        "department": str(address["department"]),
        "postal_code": str(address["postal_code"]),
        "country": str(address["country"]),
        "landmark": _as_str(address["landmark"]),
        "address_type": str(address["address_type"]),
    }

    add_tool_trace("get_customer_default_address", {}, output)
    return output


def get_order_delivery_address(order_id):
    lookup = _get_customer_order(order_id)

    if not lookup["success"]:
        output = lookup
        add_tool_trace("get_order_delivery_address", {"order_id": order_id}, output)
        return output

    order = lookup["order_row"]
    delivery_method = _as_str(order["delivery_method"])
    if delivery_method != "home_delivery":
        output = {
            "success": False,
            "message": "Este pedido fue para recogida en punto, por lo que no tiene dirección de entrega."
        }
        add_tool_trace("get_order_delivery_address", {"order_id": order_id}, output)
        return output

    address_id = _as_int(order["address_id"])
    if address_id is None:
        output = {
            "success": False,
            "message": "No se encontró una dirección de entrega para este pedido."
        }
        add_tool_trace("get_order_delivery_address", {"order_id": order_id}, output)
        return output

    address = _first_or_none(_rows_where(addresses, lambda row: _as_int(row["address_id"]) == address_id))
    if address is None:
        output = {
            "success": False,
            "message": "No se encontró la dirección asociada a este pedido."
        }
        add_tool_trace("get_order_delivery_address", {"order_id": order_id}, output)
        return output

    output = {
        "success": True,
        "order_id": _safe_order_id(order["order_id"]),
        "address_id": address_id,
        "address_line1": str(address["address_line1"]),
        "address_line2": _as_str(address["address_line2"]),
        "city": str(address["city"]),
        "department": str(address["department"]),
        "postal_code": str(address["postal_code"]),
        "country": str(address["country"]),
        "landmark": _as_str(address["landmark"]),
        "address_type": str(address["address_type"]),
        "delivery_method": delivery_method,
    }

    add_tool_trace("get_order_delivery_address", {"order_id": order_id}, output)
    return output


def get_order_history(order_id):
    lookup = _get_customer_order(order_id)

    if not lookup["success"]:
        output = lookup
        add_tool_trace("get_order_history", {"order_id": order_id}, output)
        return output

    order_tracking = _rows_where(tracking, lambda row: _same_order_id(row["order_id"], order_id))
    if not order_tracking:
        output = {
            "success": False,
            "message": "No hay historial de tracking para este pedido."
        }
        add_tool_trace("get_order_history", {"order_id": order_id}, output)
        return output

    order_tracking = _sort_rows(order_tracking, "timestamp")
    history = []
    for row in order_tracking:
        history.append({
            "timestamp": str(row["timestamp"]),
            "status": str(row["status"]),
            "location": _as_str(row["location"]),
        })

    output = {
        "success": True,
        "order_id": _safe_order_id(lookup["order_row"]["order_id"]),
        "history": history
    }
    add_tool_trace("get_order_history", {"order_id": order_id}, output)
    return output


def get_shipment_details(order_id):
    lookup = _get_customer_order(order_id)

    if not lookup["success"]:
        output = lookup
        add_tool_trace("get_shipment_details", {"order_id": order_id}, output)
        return output

    shipment = _first_or_none(_rows_where(shipments, lambda row: _same_order_id(row["order_id"], order_id)))
    if shipment is None:
        output = {
            "success": False,
            "message": "No hay información de envío para este pedido."
        }
        add_tool_trace("get_shipment_details", {"order_id": order_id}, output)
        return output

    output = {
        "success": True,
        "order_id": _safe_order_id(lookup["order_row"]["order_id"]),
        "carrier": shipment["carrier"],
        "tracking_number": shipment["tracking_number"],
        "tracking_url": shipment["tracking_url"],
        "estimated_delivery": _as_str(shipment["estimated_delivery_date"]),
        "actual_delivery": _as_str(shipment["actual_delivery_date"]),
        "delivery_attempts": _as_int(shipment["delivery_attempts"], 0),
        "status": shipment["shipment_status"],
        "failed_reason": _as_str(shipment["failed_delivery_reason"]),
    }

    add_tool_trace("get_shipment_details", {"order_id": order_id}, output)
    return output


def search_products(query: str):
    normalized_query = _normalize_catalog_text(query)
    query_tokens = _query_tokens(query)
    brand_constraints = {
        brand for brand in ["samsung", "apple", "xiaomi", "sony", "lg", "lenovo", "hp", "asus", "huawei"]
        if brand in normalized_query
    }

    subtype_constraints = {
        "celulares": {"celular", "celulares", "telefono", "telefonos", "smartphone", "iphone", "galaxy", "movil"},
        "celular": {"celular", "celulares", "telefono", "telefonos", "smartphone", "iphone", "galaxy", "movil"},
        "telefonos": {"celular", "celulares", "telefono", "telefonos", "smartphone", "iphone", "galaxy", "movil"},
        "telefono": {"celular", "celulares", "telefono", "telefonos", "smartphone", "iphone", "galaxy", "movil"},
        "calzado": {"calzado", "zapato", "zapatos", "zapatilla", "zapatillas", "tenis", "sneaker", "sneakers"},
        "zapatos": {"calzado", "zapato", "zapatos", "zapatilla", "zapatillas", "tenis", "sneaker", "sneakers"},
        "zapato": {"calzado", "zapato", "zapatos", "zapatilla", "zapatillas", "tenis", "sneaker", "sneakers"},
        "zapatillas": {"calzado", "zapato", "zapatos", "zapatilla", "zapatillas", "tenis", "sneaker", "sneakers"},
        "ropa": {"camiseta", "camisetas", "jeans", "chaqueta", "chaquetas", "vestido", "vestidos", "sudadera", "sudaderas", "blusa", "blusas", "pantalon", "pantalones", "falda", "faldas"},
        "televisores": {"televisor", "televisores", "tv", "smart", "oled", "qled", "led"},
        "tv": {"televisor", "televisores", "tv", "smart", "oled", "qled", "led"},
        "laptops": {"laptop", "laptops", "portatil", "portatiles", "notebook", "macbook", "computador"},
        "portatiles": {"laptop", "laptops", "portatil", "portatiles", "notebook", "macbook", "computador"},
    }

    required_subtype_terms = set()
    for trigger, allowed_terms in subtype_constraints.items():
        if trigger in normalized_query.split():
            required_subtype_terms |= allowed_terms

    categories_by_id = _index_by(categories, "category_id")
    brands_by_id = _index_by(brands, "brand_id")
    stock_by_product_id = _index_by(stock, "product_id")

    ranked = []
    for product in products:
        category = categories_by_id.get(product["category_id"], {})
        brand = brands_by_id.get(product["brand_id"], {})
        normalized_brand = _normalize_catalog_text(brand.get("name", ""))

        normalized_name = _normalize_catalog_text(product["name"])
        normalized_description = _normalize_catalog_text(product["description"])
        normalized_category = _normalize_catalog_text(category.get("name", ""))
        searchable_text = " ".join([normalized_name, normalized_description, normalized_category, normalized_brand])

        if brand_constraints and normalized_brand not in brand_constraints:
            continue

        if required_subtype_terms:
            product_token_blob = set(searchable_text.split())
            if not any(term in product_token_blob or term in searchable_text for term in required_subtype_terms):
                continue

        if query_tokens:
            row_tokens = set(searchable_text.split())
            score = sum(1 for token in query_tokens if token in row_tokens or token in searchable_text)

            if normalized_name in normalized_query or normalized_query in normalized_name:
                score += 5
            if normalized_category in normalized_query or normalized_query in normalized_category:
                score += 4
            if normalized_brand and (normalized_brand in normalized_query or normalized_brand == normalized_query):
                score += 3

            if score <= 0:
                continue
        else:
            if normalized_query not in searchable_text:
                continue
            score = 1

        ranked.append((score, product, category, brand, stock_by_product_id.get(product["product_id"], {})))

    ranked.sort(key=lambda item: item[0], reverse=True)

    if not ranked:
        output = {
            "success": False,
            "message": "No se encontraron productos que coincidan con la consulta."
        }
        add_tool_trace("search_products", {"query": query}, output)
        return output

    availability_terms = [
        "disponibilidad", "disponible", "disponibles", "stock",
        "a la venta", "en venta", "que tengan stock"
    ]
    availability_only = any(term in normalized_query for term in availability_terms)

    items = []
    for _, product, category, brand, stock_row in ranked:
        stock_qty = _as_int(stock_row.get("stock_qty"), 0)
        reserved_qty = _as_int(stock_row.get("reserved_qty"), 0)
        available_qty = max(stock_qty - reserved_qty, 0)
        is_active = _as_bool(product["active"])

        if availability_only and (available_qty <= 0 or not is_active):
            continue

        items.append({
            "product_id": _as_int(product["product_id"]),
            "name": str(product["name"]),
            "brand": _as_str(brand.get("name")),
            "category": _as_str(category.get("name")),
            "price": _as_float(product["price"], 0.0),
            "available_qty": int(available_qty),
            "free_shipping": _as_bool(product["free_shipping"]),
            "is_final_sale": _as_bool(product["is_final_sale"]),
            "shipping_days": _as_int(product["shipping_days"]),
            "requires_installation": _as_bool(product["requires_installation"]),
            "active": is_active,
        })

        if len(items) == 5:
            break

    if availability_only and not items:
        output = {
            "success": False,
            "message": "No encontré productos disponibles para ese filtro en este momento."
        }
        add_tool_trace("search_products", {"query": query}, output)
        return output

    output = {
        "success": True,
        "query": query,
        "results": items
    }
    add_tool_trace("search_products", {"query": query}, output)
    return output


def get_product_warranty_info(query: str):
    query_tokens = _query_tokens(query)
    normalized_query = _normalize_catalog_text(query)
    categories_by_id = _index_by(categories, "category_id")

    ranked = []
    for product in products:
        category = categories_by_id.get(product["category_id"], {})
        normalized_name = _normalize_catalog_text(product["name"])
        normalized_category = _normalize_catalog_text(category.get("name", ""))
        blob = f"{normalized_name} {normalized_category}"
        row_tokens = set(blob.split())

        if query_tokens:
            score = sum(1 for token in query_tokens if token in row_tokens or token in blob)
            if normalized_name in normalized_query or normalized_query in normalized_name:
                score += 5
            if normalized_category in normalized_query or normalized_query in normalized_category:
                score += 3
            if score <= 0:
                continue
        else:
            if normalized_query not in blob:
                continue
            score = 1

        ranked.append((score, product, category))

    ranked.sort(key=lambda item: item[0], reverse=True)

    if not ranked:
        output = {
            "success": False,
            "message": "No encontré productos o categorías del catálogo que coincidan con esa consulta de garantía."
        }
        add_tool_trace("get_product_warranty_info", {"query": query}, output)
        return output

    items = []
    for _, product, category in ranked[:5]:
        items.append({
            "product_id": _as_int(product["product_id"]),
            "name": str(product["name"]),
            "category": str(category.get("name", "")),
            "warranty_months": _as_int(product["warranty_months"], 0),
            "return_days": _as_int(product["return_days"], 0),
            "active": _as_bool(product["active"]),
        })

    output = {
        "success": True,
        "query": query,
        "results": items,
    }
    add_tool_trace("get_product_warranty_info", {"query": query}, output)
    return output


def search_promotions(query: str):
    normalized_query = _normalize_catalog_text(query)
    promo_rows = _rows_where(promotions, lambda row: _as_bool(row["active"]))

    if not promo_rows:
        output = {
            "success": False,
            "message": "No encontré promociones activas en este momento."
        }
        add_tool_trace("search_promotions", {"query": query}, output)
        return output

    category_lookup = {str(row["category_id"]): str(row["name"]) for row in categories}
    product_lookup = {str(row["product_id"]): row for row in products}

    def _promotion_targets(row):
        names = []
        searchable_context = []

        category_ids = "" if _is_missing(row["applicable_category_ids"]) else str(row["applicable_category_ids"])
        product_ids = "" if _is_missing(row["applicable_product_ids"]) else str(row["applicable_product_ids"])

        for raw_id in category_ids.split("|"):
            raw_id = raw_id.strip()
            if raw_id.isdigit() and raw_id in category_lookup:
                category_name = category_lookup[raw_id]
                names.append(category_name)
                searchable_context.append(category_name)

        for raw_id in product_ids.split("|"):
            raw_id = raw_id.strip()
            if raw_id.isdigit() and raw_id in product_lookup:
                product = product_lookup[raw_id]
                product_name = str(product["name"])
                names.append(product_name)
                searchable_context.append(product_name)
                searchable_context.append(str(product.get("description", "")))

        return names, " ".join(searchable_context)

    ranked = []
    query_tokens = _query_tokens(query)

    for row in promo_rows:
        targets, target_context = _promotion_targets(row)
        normalized_blob = _normalize_catalog_text(
            f"{row['promotion_name']} {row['description']} {' '.join(targets)} {target_context}"
        )

        if query_tokens:
            score = sum(1 for token in query_tokens if token in normalized_blob.split() or token in normalized_blob)
            minimum_score = 2 if len(set(query_tokens)) >= 2 else 1
            if score < minimum_score:
                continue
        else:
            score = 1

        ranked.append((score, row, targets))

    if not ranked and any(term in normalized_query for term in ["promocion", "promociones", "oferta", "ofertas", "descuento", "descuentos"]):
        rebuilt_ranked = []
        for row in promo_rows:
            targets, _ = _promotion_targets(row)
            rebuilt_ranked.append((1, row, targets))
        ranked = rebuilt_ranked

    ranked.sort(key=lambda item: item[0], reverse=True)

    if not ranked:
        output = {
            "success": False,
            "message": "No encontré promociones que coincidan con esa consulta."
        }
        add_tool_trace("search_promotions", {"query": query}, output)
        return output

    items = []
    for _, row, targets in ranked[:5]:
        items.append({
            "promotion_id": _as_int(row["promotion_id"]),
            "promotion_name": str(row["promotion_name"]),
            "description": str(row["description"]),
            "discount_type": str(row["discount_type"]),
            "discount_value": _as_float(row["discount_value"], 0.0),
            "min_purchase_amount": _as_float(row["min_purchase_amount"], 0.0),
            "start_date": str(row["start_date"]),
            "end_date": str(row["end_date"]),
            "targets": targets,
        })

    output = {
        "success": True,
        "query": query,
        "results": items,
    }
    add_tool_trace("search_promotions", {"query": query}, output)
    return output


def list_product_categories():
    product_counts = {}
    for product in products:
        category_id = _as_int(product["category_id"])
        product_counts[category_id] = product_counts.get(category_id, 0) + 1

    items = []
    for row in categories:
        category_id = _as_int(row["category_id"])
        items.append({
            "category_id": category_id,
            "name": str(row["name"]),
            "product_count": int(product_counts.get(category_id, 0)),
        })

    output = {
        "success": True,
        "results": items,
    }
    add_tool_trace("list_product_categories", {}, output)
    return output


def _customer_orders(customer_id: int):
    rows = _rows_where(orders, lambda row: _as_int(row["customer_id"]) == customer_id)
    return _sort_rows(rows, "order_date", reverse=True)


def get_customer_orders_summary(limit: int = 5):
    auth = _require_authenticated_customer()

    if not auth["success"]:
        output = auth
        add_tool_trace("get_customer_orders_summary", {"limit": limit}, output)
        return output

    customer_id = auth["customer_id"]
    customer_orders = _customer_orders(customer_id)

    if not customer_orders:
        output = {
            "success": False,
            "message": "No se encontraron pedidos para el cliente autenticado."
        }
        add_tool_trace("get_customer_orders_summary", {"limit": limit}, output)
        return output

    items = []
    for row in _head(customer_orders, limit):
        items.append({
            "order_id": _safe_order_id(row["order_id"]),
            "order_date": _as_str(row["order_date"]),
            "status": str(row["status"]),
            "total_amount": _as_float(row["total_amount"], 0.0),
        })

    output = {
        "success": True,
        "customer_id": customer_id,
        "results": items
    }
    add_tool_trace("get_customer_orders_summary", {"limit": limit}, output)
    return output


def get_customer_orders_for_selection(limit: int = 10):
    auth = _require_authenticated_customer()

    if not auth["success"]:
        output = auth
        add_tool_trace("get_customer_orders_for_selection", {"limit": limit}, output)
        return output

    customer_id = auth["customer_id"]
    customer_orders = _customer_orders(customer_id)

    if not customer_orders:
        output = {
            "success": False,
            "message": "No se encontraron pedidos para el cliente autenticado."
        }
        add_tool_trace("get_customer_orders_for_selection", {"limit": limit}, output)
        return output

    items = []
    for row in _head(customer_orders, limit):
        items.append({
            "order_id": _safe_order_id(row["order_id"]),
            "order_date": _as_str(row["order_date"]),
            "status": str(row["status"]),
            "total_amount": _as_float(row["total_amount"], 0.0),
        })

    output = {
        "success": True,
        "customer_id": customer_id,
        "results": items
    }
    add_tool_trace("get_customer_orders_for_selection", {"limit": limit}, output)
    return output