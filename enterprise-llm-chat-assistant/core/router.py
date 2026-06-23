"""
router.py

Clasificador de intención para el agente legacy/híbrido.
Diseñado como guía ligera: ayuda al LLM y al fallback, pero no intenta
controlar toda la conversación de forma rígida.
"""

from __future__ import annotations

from typing import Dict, Optional

from core.guards import (
    is_public_faq_query,
    is_sensitive_query,
    normalize_text,
)


IntentResult = Dict[str, Optional[str] | bool]


def _result(intent: str, requires_auth: bool, tool: Optional[str]) -> IntentResult:
    return {
        "intent": intent,
        "requires_auth": requires_auth,
        "requires_tool": tool is not None,
        "tool": tool,
    }


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


class Router:
    """
    Clasificador de intención del usuario.
    Priorizado para routing correcto del challenge, pero sin rigidizar
    excesivamente el comportamiento conversacional.
    """

    GREETINGS = [
        "hola",
        "buenas",
        "buenos dias",
        "buenas tardes",
        "buenas noches",
        "hey",
        "holi",
    ]

    PRODUCT_HINTS = [
        "precio",
        "cuanto cuesta",
        "cuánto cuesta",
        "valor",
        "presupuesto",
        "recomiendas",
        "recomienda",
        "conviene",
        "mejor",
        "comprar",
        "compra",
        "interesado",
        "interesada",
        "me interesa",
        "quisiera",
        "stock",
        "disponible",
        "disponibilidad",
        "tienen",
        "manejan",
        "venden",
        "busco",
        "quiero",
        "mostrar",
        "muestrame",
        "muéstrame",
        "opciones",
        "referencias",
        "modelos",
        "hay",
    ]

    PRODUCT_CATALOG_TERMS = [
        "calzado",
        "ropa",
        "celulares",
        "celular",
        "telefonos",
        "telefono",
        "laptops",
        "laptop",
        "portatiles",
        "portatil",
        "televisores",
        "televisor",
        "tv",
        "electrodomesticos",
        "electrodomestico",
        "nevera",
        "neveras",
        "lavadora",
        "lavadoras",
        "microondas",
        "estufa",
        "estufas",
        "hogar",
        "deportes",
        "tecnologia",
        "electronica",
        "monitores",
        "audifonos",
        "tablet",
        "tablets",
        "juguetes",
        "bebes",
        "bebe",
        "muebles",
    ]

    PROMOTION_TERMS = [
        "promocion",
        "promociones",
        "descuento",
        "descuentos",
        "oferta",
        "ofertas",
        "rebaja",
        "rebajas",
        "cupon",
        "cupón",
    ]

    # Términos exactos de política (sin conjugaciones de verbos)
    POLICY_TERMS = [
        "politica",
        "politicas",
        "garantia",
        "garantias",
        "devolucion",
        "devoluciones",
        "reembolso",
        "reembolsos",
        "envio",
        "envios",
        "cancelacion",
        "cancelar pedido",
    ]

    # Raíces de política: matchean "devolver", "devolución", "devuelvo", etc.
    POLICY_STEMS = [
        "devol",      # devolver, devolución, devoluciones, devuelvo, devuelta
        "reembols",   # reembolso, reembolsar, reembolsable
        "garantia",   # garantía, garantías
        "garantiz",   # garantiza, garantizan
        "cambio",     # cambio, cambios
        "cambia",     # cambiar, cambiarlo
        "cancel",     # cancela, cancelo, cancelar
        "cancelar",   # cancelar, cancelación
        "cancelaci",  # cancelación
        "politica",   # política, políticas
    ]

    # Patrones flexibles de política que pueden tener palabras en el medio
    # Se usa búsqueda de raíces en el texto completo
    POLICY_VERB_ROOTS = [
        "devol",      # cubre: devolver, devolverlo, devolución, devuelto, devuelvo
        "reembols",   # cubre: reembolso, reembolsar
    ]

    # Frases de intención de política (el usuario quiere SABER cómo funciona)
    POLICY_INTENT_TERMS = [
        "como funciona",
        "cómo funciona",
        "cuanto tiempo",
        "cuánto tiempo",
        "cuantos dias",
        "cuántos días",
        "plazo",
        "plazos",
        "condicion",
        "condiciones",
        "requisito",
        "requisitos",
        "aplica",
        "aplican",
        "puedo hacer",
        "puedo pedir",
        "se puede",
        "es posible",
        "como puedo",
        "cómo puedo",
        "que pasa si",
        "qué pasa si",
        "que debo",
        "qué debo",
        "como hago",
        "cómo hago",
        "tengo derecho",
        "tengo garantia",
        "tengo garantía",
        "politica de",
        "política de",
        "cuanto se demora",
        "cuánto se demora",
        "cuanto tarda",
        "cuánto tarda",
        "cuanto demora",
        "cuánto demora",
        "cuando llega",
        "cuándo llega",
        "llegar un pedido",
        "llega un pedido",
    ]

    WARRANTY_SCOPE_TERMS = [
        "producto",
        "productos",
        "articulo",
        "articulos",
        "tecnologia",
        "tecnologicos",
        "tecnologico",
        "electronica",
        "electrodomesticos",
        "electrodomestico",
        "celulares",
        "celular",
        "laptops",
        "laptop",
        "televisores",
        "televisor",
        "tv",
    ]

    ORDER_STATUS_TERMS = [
        "estado",
        "donde esta",
        "dónde está",
        "tracking",
        "seguimiento",
        "historial",
        "pedido mas reciente",
        "pedido más reciente",
        "ultimo pedido",
        "último pedido",
        "mi pedido",
        "orden",
        "pedido",
        "mis pedidos",
    ]

    RETURNS_CASE_TERMS = [
        "devolver mi pedido",
        "devolverlo",
        "devolverla",
        "quiero hacer una devolucion",
        "quiero hacer una devolución",
        "quiero un reembolso de mi pedido",
        "cambiar ese pedido",
        "reembolso de mi pedido",
        "devolucion de mi pedido",
        "devolución de mi pedido",
        "quiero cambiar mi pedido",
    ]

    ORDER_AMOUNT_TERMS = [
        "total",
        "subtotal",
        "iva",
        "monto",
        "cuanto pague",
        "cuánto pagué",
        "cuanto pagué",
        "cuanto costo",
        "cuánto costó",
        "total pagado",
    ]

    ORDER_ITEMS_TERMS = [
        "que pedi",
        "qué pedí",
        "que ordene",
        "qué ordené",
        "que compre",
        "qué compré",
        "que llevaba",
        "que incluia",
        "qué incluía",
        "productos del pedido",
        "articulos del pedido",
        "items del pedido",
    ]

    PAYMENT_METHOD_TERMS = [
        "metodo de pago",
        "método de pago",
        "medio de pago",
        "como pague",
        "cómo pagué",
        "como pagué",
        "como lo pague",
        "cómo lo pagué",
        "con que pague",
        "con qué pagué",
        "pagué con",
        "forma de pago",
    ]

    ORDER_ADDRESS_TERMS = [
        "direccion de entrega",
        "dirección de entrega",
        "direccion del pedido",
        "dirección del pedido",
        "a que direccion",
        "a qué dirección",
        "a donde iba",
        "a dónde iba",
        "donde lo iban a entregar",
        "dónde lo iban a entregar",
        "para donde iba",
        "para dónde iba",
    ]

    CUSTOMER_ADDRESS_TERMS = [
        "mi direccion",
        "mi dirección",
        "direccion registrada",
        "dirección registrada",
        "direccion principal",
        "dirección principal",
        "direccion de mi cuenta",
        "dirección de mi cuenta",
    ]

    ORDER_FOLLOWUP_SHORT_TERMS = [
        "y el total",
        "el total",
        "y la direccion",
        "y la dirección",
        "la direccion",
        "la dirección",
        "y el metodo de pago",
        "y el método de pago",
        "el metodo de pago",
        "el método de pago",
        "y que pedi",
        "y qué pedí",
        "que pedi",
        "qué pedí",
        "y el historial",
        "el historial",
        "historial completo",
        "me regalas el historial completo",
        "como lo pague",
        "cómo lo pagué",
        "cual fue mi metodo de pago",
        "cuál fue mi método de pago",
    ]

    CATEGORY_LIST_TERMS = [
        "que categorias hay",
        "qué categorías hay",
        "que categorias tienen",
        "qué categorías tienen",
        "que categorias manejan",
        "qué categorías manejan",
        "cuales categorias hay",
        "cuáles categorías hay",
        "cuales categorias tienen",
        "cuáles categorías tienen",
        "categorias disponibles",
        "categorías disponibles",
    ]

    def _is_only_greeting(self, text: str) -> bool:
        if text in self.GREETINGS:
            return True
        return any(text == f"{token}!" for token in self.GREETINGS)

    def _has_policy_stem(self, text: str) -> bool:
        """
        Detecta raíces de palabras de política para cubrir conjugaciones y
        variantes tipográficas: devolver, devolución, devuelvo, reembolsar, etc.
        """
        return any(stem in text for stem in self.POLICY_STEMS)

    def _looks_like_policy_query(self, text: str) -> bool:
        """
        Detecta consultas sobre políticas con cobertura amplia:
        - Términos exactos de política
        - Raíces de verbos (devolver, reembolsar, cancelar, cambiar)
        - Combinación de intención (cómo puedo...) + raíz de política
        """
        # Términos exactos
        detailed_shipping_policy_terms = [
            "fines de semana",
            "festivos",
            "intentos de entrega",
            "intento de entrega",
            "no estaba en casa",
            "guia",
            "seguimiento",
            "tracking",
            "cambiar direccion",
            "modificar direccion",
            "direccion de entrega",
            "costo del envio",
            "cuanto cuesta el envio",
            "cobran envio",
            "envio gratis",
            "instalacion",
            "instalar",
            "tecnico autorizado",
            "tecnico no autorizado",
            "pierde garantia",
            "garantia se pierde",
        ]

        if _contains_any(text, detailed_shipping_policy_terms):
            return True

        if _contains_any(text, self.POLICY_TERMS):
            return True

        # Raíz de política presente en cualquier parte del texto
        if self._has_policy_stem(text):
            return True

        # Intención de consultar + referencia a envío/entrega (política de envíos)
        has_policy_intent = _contains_any(text, self.POLICY_INTENT_TERMS)
        has_shipping_ref = _contains_any(text, ["envio", "envi", "entrega", "despacho", "enviar", "llegar", "llega", "demora", "tarda", "guia", "seguimiento", "tracking", "festivos"])
        if has_policy_intent and has_shipping_ref:
            return True

        return False

    def _looks_like_return_case(self, text: str) -> bool:
        """
        Distingue entre 'quiero devolver MI PEDIDO' (caso de devolución con datos del cliente)
        vs 'cómo puedo devolver' (consulta de política).
        Solo es caso de devolución si hay referencia explícita al pedido del cliente.
        """
        return _contains_any(text, self.RETURNS_CASE_TERMS)

    def _looks_like_product_query(self, text: str) -> bool:
        """
        Solo clasifica como producto si NO hay señales de política.
        Esto evita que 'devolver un celular' caiga en productos por tener 'celular'.
        """
        # Si tiene señales de política, NO es consulta de producto
        if self._looks_like_policy_query(text):
            return False

        has_catalog_term = _contains_any(text, self.PRODUCT_CATALOG_TERMS)
        has_product_signal = _contains_any(text, self.PRODUCT_HINTS)
        return has_catalog_term and has_product_signal or has_catalog_term

    def _looks_like_warranty_catalog_query(self, text: str) -> bool:
        """
        Garantía específica de un producto del catálogo (no política general).
        Requiere 'garantia' explícita + término de producto + sin señales de política general.
        """
        has_garantia = _contains_any(text, ["garantia", "garantias"])
        has_scope = _contains_any(text, self.WARRANTY_SCOPE_TERMS)
        # Si también tiene señales de política general, va a policy
        has_policy_general = _contains_any(text, ["politica", "politicas", "como funciona", "cómo funciona", "cuanto tiempo", "plazo"])
        return has_garantia and has_scope and not has_policy_general

    def _looks_like_order_amount_query(self, text: str) -> bool:
        return _contains_any(text, self.ORDER_AMOUNT_TERMS)

    def _looks_like_order_items_query(self, text: str) -> bool:
        return _contains_any(text, self.ORDER_ITEMS_TERMS)

    def _looks_like_payment_method_query(self, text: str) -> bool:
        return _contains_any(text, self.PAYMENT_METHOD_TERMS)

    def _looks_like_order_delivery_address_query(self, text: str) -> bool:
        return _contains_any(text, self.ORDER_ADDRESS_TERMS)

    def _looks_like_customer_address_query(self, text: str) -> bool:
        return _contains_any(text, self.CUSTOMER_ADDRESS_TERMS)

    def _looks_like_order_status_query(self, text: str) -> bool:
        return _contains_any(text, self.ORDER_STATUS_TERMS)

    def _looks_like_order_followup_short_query(self, text: str) -> bool:
        return _contains_any(text, self.ORDER_FOLLOWUP_SHORT_TERMS)

    def _looks_like_category_list_query(self, text: str) -> bool:
        return _contains_any(text, self.CATEGORY_LIST_TERMS)

    def classify(self, user_text: str) -> IntentResult:
        text = normalize_text(user_text)

        if not text:
            return _result("greeting", False, None)

        if self._is_only_greeting(text):
            return _result("greeting", False, None)

        # FAQs públicas simples (sin tocar políticas)
        if is_public_faq_query(text):
            return _result("faq", False, None)

        # --- POLÍTICA: máxima prioridad antes de cualquier clasificación de producto ---
        # Esto garantiza que "cómo devolver un celular", "como puedo devolver",
        # "quiero cambiar mi compra", etc. NO caigan en producto por tener
        # un término de catálogo en el texto.
        if self._looks_like_policy_query(text) and not is_sensitive_query(text):
            return _result("policy", False, "search_policy_sections")

        # Listado de categorías
        if self._looks_like_category_list_query(text):
            return _result("catalog_categories", False, "list_product_categories")

        # Garantía específica de producto/categoría (no política general)
        if self._looks_like_warranty_catalog_query(text):
            return _result("product_warranty", False, "get_product_warranty_info")

        # Casos sensibles: datos del cliente/pedido específico
        if is_sensitive_query(text):
            if self._looks_like_order_items_query(text):
                return _result("order_items", True, "get_order_items")

            if self._looks_like_order_amount_query(text):
                return _result("order_amount", True, "get_order_amounts")

            if self._looks_like_payment_method_query(text):
                return _result("order_payment_method", True, "get_order_payment_method")

            if self._looks_like_order_delivery_address_query(text):
                return _result("order_delivery_address", True, "get_order_delivery_address")

            if self._looks_like_customer_address_query(text):
                return _result("customer_address", True, "get_customer_default_address")

            if self._looks_like_return_case(text):
                return _result("returns_order_case", True, "get_order_history")

            if self._looks_like_order_status_query(text) or self._looks_like_order_followup_short_query(text):
                return _result("order_status_history", True, "get_order_status")

            if _contains_any(text, ["pedido", "orden", "mi compra", "mi orden"]):
                return _result("order_status_history", True, "get_order_status")

        # Promociones antes de productos
        if _contains_any(text, self.PROMOTION_TERMS):
            return _result("promotion_general", False, "search_promotions")

        # Productos / catálogo
        if self._looks_like_product_query(text):
            return _result("product_general", False, "search_products")

        return _result("faq", False, None)


def classify_intent(user_text: str) -> IntentResult:
    return Router().classify(user_text)
