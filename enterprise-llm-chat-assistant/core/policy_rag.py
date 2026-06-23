import os
import re
import unicodedata
from typing import List, Dict, Any, Optional

from core.session_context import add_tool_trace

BASE_POLICY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "policies")


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", (text or "").lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _read_markdown_files() -> List[Dict[str, str]]:
    documents = []

    if not os.path.isdir(BASE_POLICY_PATH):
        return documents

    for filename in os.listdir(BASE_POLICY_PATH):
        if filename.lower().endswith(".md"):
            full_path = os.path.join(BASE_POLICY_PATH, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            documents.append({
                "source": filename,
                "content": content,
            })

    return documents


def _split_markdown_by_headers(source: str, content: str) -> List[Dict[str, Any]]:
    lines = content.splitlines()
    sections = []

    current_header = "Documento completo"
    current_lines = []

    for line in lines:
        if re.match(r"^\s*#{1,6}\s+", line):
            if current_lines:
                text = "\n".join(current_lines).strip()
                if text:
                    sections.append({
                        "source": source,
                        "header": current_header,
                        "text": text,
                    })
            current_header = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        text = "\n".join(current_lines).strip()
        if text:
            sections.append({
                "source": source,
                "header": current_header,
                "text": text,
            })

    return sections


def _clean_markdown_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\\r\\n", "\n")
    text = text.replace("\\n", "\n")
    text = text.replace("\\t", "\t")

    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)

    lines = [line.strip() for line in text.splitlines()]

    cleaned_lines = []
    previous_blank = False
    for line in lines:
        is_blank = (line == "")
        if is_blank and previous_blank:
            continue
        cleaned_lines.append(line)
        previous_blank = is_blank

    return "\n".join(cleaned_lines).strip()


def _extract_lines(text: str) -> List[str]:
    cleaned = _clean_markdown_text(text)
    if not cleaned:
        return []
    return [line.strip() for line in cleaned.splitlines() if line.strip()]


def _extract_numbered_steps(text: str) -> List[str]:
    steps = []
    for line in _extract_lines(text):
        match = re.match(r"^\d+\.\s*(.+)$", line)
        if match:
            steps.append(match.group(1).strip().rstrip("."))
    return steps


def _extract_bullets(text: str) -> List[str]:
    bullets = []
    for line in _extract_lines(text):
        if line.startswith("* ") or line.startswith("- "):
            bullets.append(line[2:].strip().rstrip("."))
    return bullets


def _extract_markdown_table(text: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for raw_line in _clean_markdown_text(text).splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells:
            continue
        if all(re.fullmatch(r"[-:\s]+", cell) for cell in cells):
            continue
        rows.append(cells)
    return rows


def _extract_faq_entries(text: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    current_question: Optional[str] = None
    current_answer: List[str] = []

    for raw_line in _clean_markdown_text(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        is_question = line.endswith("?") or (line.startswith("¿") and "?" in line)
        if is_question:
            if current_question and current_answer:
                entries.append({
                    "question": current_question,
                    "answer_lines": current_answer[:],
                })
            current_question = line
            current_answer = []
            continue

        if current_question:
            current_answer.append(line.lstrip("*- ").strip())

    if current_question and current_answer:
        entries.append({
            "question": current_question,
            "answer_lines": current_answer[:],
        })

    return entries


def _faq_score(query: str, question: str) -> int:
    query_tokens = set(_tokenize(query))
    question_tokens = set(_tokenize(question))
    overlap = len(query_tokens & question_tokens)
    stem_overlap = _stem_score(_extract_policy_stems(query), question) * 2

    normalized_query = _normalize(query)
    normalized_question = _normalize(question)
    exact_bonus = 6 if normalized_query and normalized_query in normalized_question else 0
    return overlap * 4 + stem_overlap + exact_bonus


def _find_best_faq_entry(sections: List[Dict[str, Any]], query: str) -> Optional[Dict[str, Any]]:
    best_entry: Optional[Dict[str, Any]] = None
    best_score = 0

    for section in sections:
        header = _normalize(section.get("title", ""))
        if "preguntas frecuentes" not in header and "faq" not in header:
            continue

        for entry in _extract_faq_entries(section.get("content", "")):
            score = _faq_score(query, entry.get("question", ""))
            if score > best_score:
                best_score = score
                best_entry = entry

    if best_score >= 6:
        return best_entry
    return None


def _join_answer_lines(answer_lines: List[str]) -> str:
    cleaned = [line.strip().rstrip(".") for line in answer_lines if line and line.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0] + "."
    return "; ".join(cleaned) + "."


def _first_sentence(text: str) -> str:
    normalized = " ".join(_extract_lines(text))
    if not normalized:
        return ""
    match = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)
    return match[0].strip()


def _find_section(sections: List[Dict[str, Any]], *header_terms: str) -> Optional[Dict[str, Any]]:
    for section in sections:
        header = _normalize(section.get("title", ""))
        if all(term in header for term in header_terms):
            return section
    return None


def _find_section_any(sections: List[Dict[str, Any]], header_options: List[str]) -> Optional[Dict[str, Any]]:
    for option in header_options:
        section = _find_section(sections, *option.split())
        if section is not None:
            return section
    return None


def _join_phrases(parts: List[str]) -> str:
    cleaned = [part.strip() for part in parts if part and part.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} y {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])} y {cleaned[-1]}"


def _lowercase_first(text: str) -> str:
    if not text:
        return ""
    return text[0].lower() + text[1:]


def _detect_returns_category_from_query(query: str) -> Optional[str]:
    normalized = _normalize(query)
    category_aliases = {
        "Ropa y calzado": [
            "ropa",
            "calzado",
            "zapatilla",
            "zapatillas",
            "zapato",
            "zapatos",
            "tenis",
            "sneaker",
            "sneakers",
            "nike",
            "adidas",
            "puma",
        ],
        "Electrónica": [
            "celular",
            "celulares",
            "telefono",
            "telefonos",
            "iphone",
            "samsung",
            "xiaomi",
            "tablet",
            "tablets",
            "monitor",
            "monitores",
            "audifono",
            "audifonos",
            "electronica",
        ],
        "Electrodomésticos": [
            "electrodomesticos",
            "electrodomestico",
            "nevera",
            "lavadora",
            "estufa",
            "licuadora",
            "microondas",
            "cafetera",
            "aire acondicionado",
        ],
        "Hogar y muebles": [
            "hogar",
            "mueble",
            "muebles",
            "sofa",
            "sofá",
            "cama",
            "mesa",
            "escritorio",
            "colchon",
            "colchón",
        ],
        "Juguetes y bebés": [
            "juguete",
            "juguetes",
            "bebe",
            "bebes",
            "bebé",
            "bebés",
        ],
    }

    for category, aliases in category_aliases.items():
        if any(alias in normalized for alias in aliases):
            return category
    return None


def _is_returns_deadline_query(query: str) -> bool:
    normalized = _normalize(query)
    return any(
        term in normalized
        for term in [
            "plazo",
            "plazos",
            "cuantos dias",
            "cuanto tiempo",
            "tiempo para devolver",
            "condiciones",
            "puedo devolver",
            "se puede devolver",
            "aplica devolucion",
        ]
    )


def _extract_deadline_table_details(
    deadline_section: Optional[Dict[str, Any]],
    requested_category: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not deadline_section:
        return None

    table = _extract_markdown_table(deadline_section.get("content", ""))
    if len(table) <= 1:
        return None

    rows = [row for row in table[1:] if len(row) >= 3]
    if not rows:
        return None

    if requested_category:
        for row in rows:
            if _normalize(row[0]) == _normalize(requested_category):
                return {
                    "mode": "category",
                    "category": row[0],
                    "deadline": row[1],
                    "condition": row[2].rstrip("."),
                }

    common_deadlines = {row[1] for row in rows if row[1]}
    categories = [row[0] for row in rows if row[0]]
    return {
        "mode": "general",
        "deadline": next(iter(common_deadlines)) if len(common_deadlines) == 1 else None,
        "categories": categories,
        "examples": [
            {
                "category": row[0],
                "deadline": row[1],
                "condition": row[2].rstrip("."),
            }
            for row in rows[:4]
        ],
    }


def _condition_phrase(condition: str) -> str:
    cleaned = (condition or "").strip().rstrip(".")
    if not cleaned:
        return ""

    normalized = _normalize(cleaned)
    if normalized.startswith("producto "):
        remainder = cleaned.split(" ", 1)[1].strip() if " " in cleaned else cleaned
        return f"esté {remainder[0].lower() + remainder[1:]}" if remainder else "esté en buen estado"
    if normalized.startswith("productos "):
        remainder = cleaned.split(" ", 1)[1].strip() if " " in cleaned else cleaned
        return f"sean {remainder[0].lower() + remainder[1:]}" if remainder else "sean aptos para devolución"
    return f"esté {cleaned[0].lower() + cleaned[1:]}"


def _detect_policy_subtopic(query: str, topic: Optional[str]) -> Optional[str]:
    q = _normalize(query)

    if topic == "shipping":
        if any(term in q for term in ["fines de semana", "fin de semana", "festivos", "dias festivos"]):
            return "weekend_schedule"
        if any(term in q for term in ["cancelar", "cancel", "antes del despacho", "antes de despachar"]):
            return "cancellation"
        if any(term in q for term in ["cambiar direccion", "cambiar direccion de envio", "modificar direccion", "direccion", "en camino"]):
            return "address_change"
        if any(term in q for term in ["cuanto se demora", "cuanto tarda", "cuando llega", "tiempo de entrega", "demora", "tarda", "llega"]):
            return "delivery_time"
        if any(term in q for term in ["tracking", "guia", "seguimiento", "transportadora", "numero de guia", "mandan la guia", "mandan guia"]):
            return "tracking"
        if any(term in q for term in ["intento de entrega", "intentos de entrega", "no estaba en casa", "no me encontraba", "no habia nadie", "no habia quien recibiera"]):
            return "delivery_attempts"
        if any(term in q for term in ["costo de envio", "cuanto cuesta el envio", "envio gratis", "cobran envio", "cobra envio", "tiene costo el envio"]):
            return "shipping_cost"

    if topic == "returns":
        if any(term in q for term in ["descuento", "promocion", "promoción", "venta final", "final sale"]):
            return "discount_items"
        if any(term in q for term in ["cancelar", "cancel", "antes del despacho", "antes de despachar", "en camino"]):
            return "cancellation"
        if any(term in q for term in ["reembolso", "devolver dinero", "reembols"]):
            return "refund"
        if any(term in q for term in ["cambio", "cambiar producto", "cambiar un producto", "por otro producto", "cambiarlo por otro"]):
            return "exchange"
        if any(term in q for term in ["hay costo", "costo", "cuesta", "cargo", "cobran", "cobro", "envio original"]):
            return "cost"
        if any(term in q for term in ["como devuelvo", "como hago la devolucion", "solicitar devolucion", "solicitar cambio", "como solicito", "como inicio", "iniciar devolucion"]):
            return "process"
        if any(term in q for term in ["plazo", "cuantos dias", "cuánto tiempo", "tiempo para devolver"]):
            return "deadline"

    if topic == "warranty":
        if any(term in q for term in ["no cubre", "exclusion", "exclusiones", "tecnico no autorizado", "tecnico autorizado", "personal no autorizado", "pierde garantia", "pierde la garantia", "garantia se pierde", "se pierde la garantia", "anula garantia", "anula la garantia", "instalacion", "instalar", "mal uso", "golpes", "caidas", "humedad", "danos por agua", "dano por agua"]):
            return "exclusions"
        if any(term in q for term in ["que cubre", "qué cubre", "cubre la garantia"]):
            return "coverage"
        if any(term in q for term in ["no cubre", "exclusion", "exclusiones", "tecnico no autorizado", "personal no autorizado", "pierde garantia", "pierde la garantia", "anula garantia", "anula la garantia", "mal uso", "golpes", "caidas", "humedad", "danos por agua", "dano por agua"]):
            return "exclusions"
        if any(term in q for term in ["como pido", "como reclamo", "solicitar garantia", "proceso", "como solicito", "como inicio", "iniciar garantia", "reclamar garantia", "inicio un reclamo"]):
            return "process"
        if any(term in q for term in ["cuanto dura", "vigencia", "cuanto tiempo", "cuántos meses"]):
            return "term"

    return None


def _format_returns_policy_response(result: Dict[str, Any], sections: List[Dict[str, Any]]) -> str:
    query = result.get("query", "")
    subtopic = _detect_policy_subtopic(query, "returns")
    requested_category = _detect_returns_category_from_query(query)
    faq_entry = _find_best_faq_entry(sections, query)
    if subtopic == "cancellation":
        intro = "Claro. Sobre la cancelación del pedido, esto es lo que aplica."
    elif subtopic == "discount_items":
        intro = "Claro. Sobre productos con descuento o promoción, esto es lo importante."
    elif requested_category:
        intro = f"Claro. Si lo que quieres es devolver algo de {requested_category.lower()}, te explico cómo funciona."
    else:
        intro = "Claro. Si lo que necesitas es devolver algo, te explico cómo funciona."
    process_section = _find_section(sections, "proceso") or _find_section(sections, "solicitar")
    deadline_section = _find_section(sections, "plazos") or _find_section(sections, "devoluciones")
    conditions_section = _find_section(sections, "condiciones")
    deadline_details = _extract_deadline_table_details(deadline_section, requested_category)

    paragraphs = [intro]

    if subtopic == "discount_items":
        ineligible_section = _find_section_any(sections, ["articulos no elegibles para devolucion"])
        if ineligible_section:
            lines = _extract_lines(ineligible_section.get("content", ""))
            special_line = next((item for item in lines if "promocion" in _normalize(item)), None)
            if special_line is None:
                special_line = next((item for item in lines if "venta final" in _normalize(item)), None)
            if special_line:
                paragraphs.append("En ese caso hay una restriccion importante.")
                paragraphs.append(special_line.rstrip(".") + ".")
                return "\n\n".join(paragraphs)

    if _is_returns_deadline_query(query) and deadline_details:
        if deadline_details["mode"] == "category":
            paragraphs.append(
                f"Para {deadline_details['category'].lower()}, el plazo es de {deadline_details['deadline']}."
            )
            paragraphs.append(
                f"La condición principal es que {_condition_phrase(deadline_details['condition'])}."
            )
            paragraphs.append(
                "Ten presente que algunos productos pueden tener un plazo distinto o no tener devolución, así que conviene revisar la ficha del producto."
            )
            return "\n\n".join(paragraphs)

        if deadline_details["mode"] == "general":
            if deadline_details.get("deadline"):
                paragraphs.append(
                    f"En esa sección, el plazo general que aparece es de {deadline_details['deadline']} para categorías como "
                    f"{_join_phrases(deadline_details['categories'][:5])}."
                )
            examples = deadline_details.get("examples", [])
            if examples:
                example_lines = [
                    f"{item['category']}: {item['condition'].lower()}"
                    for item in examples[:3]
                ]
                paragraphs.append(
                    "Las condiciones cambian según la categoría; por ejemplo, "
                    + "; ".join(example_lines)
                    + "."
                )
            paragraphs.append(
                "Además, la política aclara que algunos productos pueden tener un plazo distinto o no admitir devolución."
            )
            return "\n\n".join(paragraphs)

    if faq_entry and subtopic in {"deadline", "process"} and not requested_category:
        faq_answer = _join_answer_lines(faq_entry.get("answer_lines", []))
        if faq_answer:
            paragraphs.append(faq_answer)
            paragraphs.append("Si quieres, también te digo si en tu caso te conviene más cambio, crédito o reembolso.")
            return "\n\n".join(paragraphs)

    if subtopic == "refund":
        refund_section = _find_section_any(sections, ["plazos de reembolso", "reembolso al medio de pago"])
        if refund_section:
            table = _extract_markdown_table(refund_section.get("content", ""))
            if len(table) > 1:
                examples = [
                    f"{row[0]}: {row[1]}"
                    for row in table[1:5]
                    if len(row) >= 2 and row[0] and row[1]
                ]
                paragraphs.append("Si pides reembolso, el dinero vuelve al medio de pago original.")
                if examples:
                    paragraphs.append("Los tiempos varían según el medio de pago; por ejemplo, " + "; ".join(examples) + ".")
                paragraphs.append("Además, el envío original no suele ser reembolsable.")
                return "\n\n".join(paragraphs)

    if subtopic == "cost":
        paragraphs.append("Para cambios o credito en tienda, la politica indica que no suele haber cargo adicional por el envio de devolucion.")
        paragraphs.append("Si eliges reembolso al medio de pago, puede aplicar un cargo administrativo y el envio original no suele ser reembolsable.")
        return "\n\n".join(paragraphs)

    if subtopic == "exchange":
        exchange_section = _find_section_any(sections, ["cambio por otro producto", "opciones de devolucion"])
        if exchange_section:
            paragraphs.append(
                "Sí, puedes pedir cambio por otro producto disponible, por ejemplo por talla, color, modelo o referencia."
            )
            paragraphs.append(
                "Normalmente queda sujeto a disponibilidad al momento en que procesan la devolución."
            )
            return "\n\n".join(paragraphs)

    if subtopic == "discount_items":
        ineligible_section = _find_section_any(sections, ["articulos no elegibles para devolucion"])
        if ineligible_section:
            lines = _extract_lines(ineligible_section.get("content", ""))
            special_line = next((item for item in lines if "promocion" in _normalize(item)), None)
            if special_line is None:
                special_line = next((item for item in lines if "venta final" in _normalize(item)), None)
            if special_line:
                paragraphs.append("En ese caso hay una restricción importante.")
                paragraphs.append(special_line.rstrip(".") + ".")
                return "\n\n".join(paragraphs)

    if subtopic == "cancellation":
        before_dispatch = _find_section_any(sections, ["antes del despacho"])
        after_dispatch = _find_section_any(sections, ["despues del despacho", "después del despacho"])
        if before_dispatch or after_dispatch:
            if before_dispatch:
                paragraphs.append(
                    "Si el pedido todavía no ha sido despachado, normalmente sí puedes cancelarlo desde Mis Pedidos cuando esté como solicitud recibida o en preparación."
                )
            if after_dispatch:
                paragraphs.append(
                    "Si ya va en camino o ya fue despachado, ya no se puede cancelar y te tocaría recibirlo y luego pedir la devolución."
                )
            return "\n\n".join(paragraphs)

    if process_section:
        steps = _extract_numbered_steps(process_section.get("content", ""))
        if steps:
            useful_steps = [_lowercase_first(step) for step in steps[:5]]
            paragraphs.append(
                "Normalmente la solicitud se hace desde Mi Cuenta > Mis Pedidos: "
                + ", luego ".join(
                    [
                        useful_steps[0],
                        useful_steps[1] if len(useful_steps) > 1 else "",
                        useful_steps[2] if len(useful_steps) > 2 else "",
                    ]
                ).replace(", luego ,", ", ")
                + "."
            )
            if len(useful_steps) > 3:
                paragraphs.append(
                    "Después indicas el motivo de la devolución y escoges si prefieres cambio, crédito en tienda o reembolso."
                )
            paragraphs.append("Al final te generan un número de caso para hacer seguimiento.")
        else:
            sentence = _first_sentence(process_section.get("content", ""))
            if sentence:
                paragraphs.append(sentence)

    if deadline_section:
        table = _extract_markdown_table(deadline_section.get("content", ""))
        if len(table) > 1:
            data_rows = [row for row in table[1:] if len(row) >= 3][:5]
            if requested_category:
                matching_rows = [
                    row for row in table[1:]
                    if len(row) >= 3 and _normalize(row[0]) == _normalize(requested_category)
                ]
                if matching_rows:
                    row = matching_rows[0]
                    paragraphs.append(
                        f"Para {row[0].lower()}, el plazo que veo en la política es de {row[1]}, "
                        f"y la condición principal es: {row[2].rstrip('.')}."
                    )
                    data_rows = []
            categories = [row[0] for row in data_rows if row and row[0]]
            common_deadlines = {row[1] for row in data_rows if len(row) >= 2 and row[1]}
            if categories and len(common_deadlines) == 1:
                paragraphs.append(
                    f"En la política que encontré, el plazo general es de {next(iter(common_deadlines))} para categorías como "
                    f"{_join_phrases(categories[:4])}, siempre que el producto esté en buenas condiciones."
                )
            elif data_rows:
                examples = [
                    f"{row[0]}: {row[1]}"
                    for row in data_rows[:3]
                    if len(row) >= 2 and row[0] and row[1]
                ]
                if examples:
                    paragraphs.append(
                        "El plazo depende de la categoría. Por ejemplo, "
                        + "; ".join(examples)
                        + "."
                    )

    if conditions_section:
        bullets = _extract_bullets(conditions_section.get("content", ""))
        if bullets:
            paragraphs.append(
                "Para que te la acepten, normalmente piden que el producto esté "
                + _join_phrases(bullets[:3]).lower()
                + "."
            )

    paragraphs.append("Si quieres, también te digo si en tu caso te conviene más cambio, crédito o reembolso.")
    return "\n\n".join(paragraphs)


def _format_warranty_policy_response(result: Dict[str, Any], sections: List[Dict[str, Any]]) -> str:
    query = result.get("query", "")
    subtopic = _detect_policy_subtopic(query, "warranty")
    faq_entry = _find_best_faq_entry(sections, query)
    if subtopic == "coverage":
        intro = "Claro. Esto es lo que sí cubre la garantía."
    elif subtopic == "exclusions":
        intro = "Claro. Esto es lo que normalmente no cubre la garantía."
    elif subtopic == "process":
        intro = "Claro. Así puedes iniciar un reclamo de garantía."
    else:
        intro = "Claro. Si es un tema de garantía, la cobertura aplica cuando el producto tiene una falla de funcionamiento o un defecto de fabricación."
    coverage_section = _find_section(sections, "cubre") or _find_section(sections, "cobertura")
    process_section = _find_section(sections, "reclamacion") or _find_section(sections, "reclamación")
    vigencia_section = _find_section(sections, "vigencia") or _find_section(sections, "garantia por categoria")
    timing_section = _find_section(sections, "plazos") or vigencia_section

    paragraphs = [intro]

    if subtopic == "coverage":
        if faq_entry:
            faq_answer = _join_answer_lines(faq_entry.get("answer_lines", []))
            if faq_answer:
                paragraphs.append(faq_answer)
                return "\n\n".join(paragraphs)
        faq_section = _find_section_any(sections, ["preguntas frecuentes", "cobertura general"])
        if faq_section:
            cleaned = _clean_markdown_text(faq_section.get("content", ""))
            match = re.search(r"Qué cubre la garantía\?\s*(.+?)(?:\n\n|¿Qué NO cubre|\Z)", cleaned, flags=re.IGNORECASE | re.DOTALL)
            if match:
                paragraphs.append(match.group(1).replace("\n", " ").strip())
                return "\n\n".join(paragraphs)

    if subtopic == "exclusions":
        if any(term in _normalize(query) for term in ["instalacion", "instalar", "tecnico autorizado"]):
            paragraphs.append("Para productos que requieren instalacion, la politica indica que debe hacerla un tecnico autorizado de la marca; de lo contrario la garantia puede perderse.")
            return "\n\n".join(paragraphs)
        if any(term in _normalize(query) for term in ["tecnico no autorizado", "personal no autorizado", "pierde garantia", "pierde la garantia", "anula garantia", "anula la garantia"]):
            paragraphs.append("Si interviene un tecnico no autorizado, la politica indica que la garantia puede perderse o quedar anulada.")
            return "\n\n".join(paragraphs)
        if faq_entry:
            faq_question = _normalize(faq_entry.get("question", ""))
            faq_answer = _join_answer_lines(faq_entry.get("answer_lines", []))
            if faq_answer and any(marker in faq_question for marker in ["no cubre", "que no cubre", "qué no cubre"]):
                paragraphs.append(faq_answer)
                return "\n\n".join(paragraphs)
        coverage_section = _find_section_any(sections, ["no cubierto", "exclusiones especificas por categoria"])
        if coverage_section:
            cleaned_exclusions = _clean_markdown_text(coverage_section.get("content", ""))
            if any(term in _normalize(query) for term in ["tecnico no autorizado", "personal no autorizado", "pierde garantia", "pierde la garantia", "anula garantia", "anula la garantia"]) and "no autorizado" in _normalize(cleaned_exclusions):
                paragraphs.append("Si interviene un tecnico no autorizado, la politica indica que la garantia puede perderse o quedar anulada.")
                return "\n\n".join(paragraphs)
            bullets = _extract_bullets(coverage_section.get("content", ""))
            if bullets:
                paragraphs.append("Lo que normalmente no cubre es " + _join_phrases(bullets[:4]).lower() + ".")
                return "\n\n".join(paragraphs)
            exclusion_lines = [line.rstrip(".") for line in _extract_lines(coverage_section.get("content", "")) if line.strip() and line.strip() != "---"]
            if exclusion_lines:
                specific_line = next(
                    (
                        line for line in exclusion_lines
                        if any(term in _normalize(query) and term in _normalize(line) for term in ["golpes", "caidas", "humedad", "agua", "voltaje", "desgaste", "consumibles"])
                    ),
                    None,
                )
                if specific_line:
                    paragraphs.append(specific_line + ".")
                    return "\n\n".join(paragraphs)
                paragraphs.append("Lo que normalmente no cubre es " + _join_phrases(exclusion_lines[:4]).lower() + ".")
                return "\n\n".join(paragraphs)

    if subtopic == "process":
        if faq_entry and any(marker in _normalize(faq_entry.get("question", "")) for marker in ["como inicio", "como solicito", "como pido", "como inicio un reclamo", "como reclamo"]):
            faq_answer = _join_answer_lines(faq_entry.get("answer_lines", []))
            if faq_answer:
                paragraphs.append(faq_answer)
                paragraphs.append("Después revisan el caso y te indican si aplica reparación, cambio o reembolso.")
                return "\n\n".join(paragraphs)
        step_section = _find_section_any(sections, ["paso 2", "paso 1", "paso 3"])
        if step_section:
            numbered_steps = _extract_numbered_steps(step_section.get("content", ""))
            if numbered_steps:
                useful_steps = [_lowercase_first(step) for step in numbered_steps[:4]]
                paragraphs.append(
                    "Para pedir la garantia, normalmente " + ", luego ".join(useful_steps[:3]) + "."
                )
                paragraphs.append("Despues revisan el caso y te indican si aplica reparacion, cambio o reembolso.")
                return "\n\n".join(paragraphs)
        process_section = _find_section_any(sections, ["proceso de reclamacion de garantia", "si tu producto presenta fallas de funcionamiento"])
        if process_section:
            steps = _extract_numbered_steps(process_section.get("content", ""))
            if steps:
                useful_steps = [_lowercase_first(step) for step in steps[:3]]
                paragraphs.append(
                    "Para pedir la garantia, normalmente " + ", luego ".join(useful_steps) + "."
                )
                paragraphs.append("Despues revisan el caso y te indican si aplica reparacion, cambio o reembolso.")
                return "\n\n".join(paragraphs)
            paragraphs.append(
                "Para pedir la garantía, normalmente entras a tu cuenta, buscas el pedido, eliges “Solicitar Garantía”, describes el problema y adjuntas evidencia si la tienes."
            )
            paragraphs.append("Después revisan el caso y te indican si aplica reparación, cambio o reembolso.")
            return "\n\n".join(paragraphs)

    if subtopic == "term" and vigencia_section:
        table = _extract_markdown_table(vigencia_section.get("content", ""))
        if len(table) > 1:
            examples = [
                f"{row[0]}: {row[1]}"
                for row in table[1:5]
                if len(row) >= 2 and row[0] and row[1]
            ]
            if examples:
                paragraphs.append(
                    "La vigencia cambia segun la categoria; por ejemplo, "
                    + "; ".join(examples)
                    + "."
                )
                paragraphs.append("El periodo exacto de cada producto se confirma en la ficha del producto.")
                return "\n\n".join(paragraphs)

    if coverage_section:
        bullets = _extract_bullets(coverage_section.get("content", ""))
        if bullets:
            paragraphs.append(
                "En general cubre "
                + _join_phrases(bullets[:3]).lower()
                + "."
            )

    if timing_section:
        table = _extract_markdown_table(timing_section.get("content", ""))
        if len(table) > 1:
            examples = [
                f"{row[0]}: {row[1]}"
                for row in table[1:4]
                if len(row) >= 2 and row[0] and row[1]
            ]
            if examples:
                paragraphs.append(
                    "El tiempo de garantía cambia según la categoría; por ejemplo, "
                    + "; ".join(examples)
                    + "."
                )

    if process_section:
        steps = _extract_numbered_steps(process_section.get("content", ""))
        if steps:
            paragraphs.append(
                "Para iniciarla, normalmente entras a tu cuenta, buscas el pedido, eliges “Solicitar Garantía” y describes el problema."
            )

    paragraphs.append("Si me dices qué producto es o qué falla tiene, te lo aterrizo mejor.")
    return "\n\n".join(paragraphs)


def _format_shipping_policy_response(result: Dict[str, Any], sections: List[Dict[str, Any]]) -> str:
    query = result.get("query", "")
    subtopic = _detect_policy_subtopic(query, "shipping")
    faq_entry = _find_best_faq_entry(sections, query)
    if subtopic == "delivery_time":
        intro = "Claro. Esto es lo que dice la política sobre los tiempos de entrega."
    elif subtopic == "weekend_schedule":
        intro = "Claro. Sobre fines de semana y festivos, esto es lo que aplica."
    elif subtopic == "address_change":
        intro = "Claro. Sobre cambios de dirección, esto es lo que aplica."
    elif subtopic == "tracking":
        intro = "Claro. Sobre el seguimiento del pedido, esto es lo que aplica."
    elif subtopic == "delivery_attempts":
        intro = "Claro. Sobre los intentos de entrega, esto es lo que aplica."
    elif subtopic == "shipping_cost":
        intro = "Claro. Sobre el costo del envío, esto es lo que encontré."
    else:
        intro = "Claro. Te cuento cómo manejan los envíos."
    processing_section = _find_section(sections, "procesamiento")
    coverage_section = _find_section(sections, "cobertura")
    delivery_section = _find_section(sections, "tiempos") or _find_section(sections, "entrega")
    tracking_section = _find_section(sections, "seguimiento")

    paragraphs = [intro]

    if faq_entry and subtopic in {"delivery_time", "tracking"}:
        faq_answer = _join_answer_lines(faq_entry.get("answer_lines", []))
        if faq_answer:
            paragraphs.append(faq_answer)
            if subtopic == "delivery_time":
                paragraphs.append("Ten presente que esos tiempos son estimados y pueden cambiar por clima, temporada alta o la ubicación.")
            return "\n\n".join(paragraphs)

    if subtopic == "weekend_schedule" and processing_section:
        cleaned_processing = _clean_markdown_text(processing_section.get("content", ""))
        if "fines de semana" in _normalize(cleaned_processing) or "festivos" in _normalize(cleaned_processing):
            paragraphs.append("La politica indica que no realizan envios ni entregas los fines de semana ni los dias festivos.")
            paragraphs.append("Los tiempos publicados se cuentan en dias habiles.")
            return "\n\n".join(paragraphs)

    if subtopic == "address_change":
        address_section = _find_section_any(sections, ["modificacion de direccion de envio", "reprogramacion de entregas"])
        if address_section:
            cleaned = _clean_markdown_text(address_section.get("content", ""))
            if "antes del despacho" in _normalize(cleaned) and "despues del despacho" in _normalize(cleaned):
                paragraphs.append("Si el pedido todavía no ha sido despachado, sí puedes pedir el cambio de dirección contactando soporte.")
                paragraphs.append("Si ya aparece en camino, la política dice que no se puede garantizar ese cambio.")
                return "\n\n".join(paragraphs)

    if subtopic == "cancellation":
        cancellation_section = _find_section_any(sections, ["cancelacion de pedidos"])
        if cancellation_section:
            paragraphs.append("Si el pedido todavía no ha sido despachado, normalmente sí puedes cancelarlo desde Mis Pedidos cuando esté como solicitud recibida o en preparación.")
            paragraphs.append("Si ya fue despachado o aparece en camino, ya no se puede cancelar y tendrías que pedir la devolución después.")
            return "\n\n".join(paragraphs)

    if subtopic == "tracking":
        tracking_section = _find_section_any(sections, ["seguimiento de pedidos", "informacion de tracking"])
        if tracking_section:
            paragraphs.append(
                "Una vez despachan el pedido, normalmente te llega por correo la guía, el enlace de seguimiento y la información de la transportadora."
            )
            paragraphs.append("Ten presente que la actualización del tracking puede tardar entre 24 y 48 horas después del despacho.")
            return "\n\n".join(paragraphs)

    if subtopic == "delivery_attempts":
        attempts_section = _find_section_any(sections, ["intentos de entrega"])
        if attempts_section:
            paragraphs.append("La política dice que la transportadora hace hasta 3 intentos de entrega.")
            paragraphs.append("Si no logran entregarlo, el paquete vuelve al centro de distribución y luego coordinan un nuevo envío.")
            return "\n\n".join(paragraphs)

    if subtopic == "shipping_cost":
        shipping_cost_section = _find_section_any(sections, ["costos de envio", "envio gratuito", "tarifas estandar"])
        if shipping_cost_section:
            paragraphs.append("El costo del envio se calcula automaticamente antes de confirmar la compra.")
            paragraphs.append("Depende de la ubicacion de entrega, el peso, las dimensiones y el metodo de envio.")
            paragraphs.append("Algunos productos o promociones pueden tener envio gratis.")
            return "\n\n".join(paragraphs)

    if processing_section:
        sentence = _first_sentence(processing_section.get("content", ""))
        if sentence:
            paragraphs.append(sentence)

    if coverage_section:
        sentence = _first_sentence(coverage_section.get("content", ""))
        if sentence:
            paragraphs.append(sentence)

    if delivery_section:
        zone_lines = []
        for line in _extract_lines(delivery_section.get("content", "")):
            normalized_line = _normalize(line)
            if any(marker in normalized_line for marker in ["ciudades principales", "ciudades intermedias", "zonas rurales", "envio estandar", "envio express"]):
                zone_lines.append(line.rstrip("."))
        if zone_lines:
            paragraphs.append("En la política aparecen estos tiempos de referencia: " + "; ".join(zone_lines[:5]) + ".")
        else:
            table = _extract_markdown_table(delivery_section.get("content", ""))
            if len(table) > 1:
                examples = [
                    f"{row[0]}: {row[1]}"
                    for row in table[1:4]
                    if len(row) >= 2 and row[0] and row[1]
                ]
                if examples:
                    paragraphs.append(
                        "Los tiempos estimados cambian según la zona; por ejemplo, "
                        + "; ".join(examples)
                        + "."
                    )
            else:
                sentence = _first_sentence(delivery_section.get("content", ""))
                if sentence:
                    paragraphs.append(sentence)

    if tracking_section:
        paragraphs.append(
            "Una vez despachan el pedido, normalmente te envían la guía, el enlace de seguimiento y la información de la transportadora."
        )

    paragraphs.append("Si quieres, también te resumo qué pasa con cambios de dirección, demoras o intentos de entrega.")
    return "\n\n".join(paragraphs)


def _tokenize(text: str) -> List[str]:
    text = _normalize(text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return [t for t in text.split() if t]


# Mapa de raíces semánticas para matching flexible.
# Clave: raíz que puede aparecer en la query (incluye typos comunes).
# Valor: lista de raíces que deben existir en el documento de política.
_STEM_SYNONYMS: Dict[str, List[str]] = {
    # raíces de devolución — cubre: devolver, devolución, devuelvo, devoluvion (typo)
    "devol": ["devol"],
    "devolv": ["devol"],
    "devoluc": ["devol"],
    "devolui": ["devol"],   # typo: devoluvion, devolui*
    # reembolso
    "reembols": ["reembols", "devol"],
    # garantía
    "garantia": ["garantia", "garant"],
    "garantiz": ["garantia", "garant"],
    # cancelar
    "cancelar": ["cancelar", "cancelaci"],
    "cancelaci": ["cancelar", "cancelaci"],
    # cambio / cambiar
    "cambio": ["cambio", "devol"],
    "cambi": ["cambio", "devol"],
    # envío / enviar
    "envio": ["envio", "envi", "despacho"],
    "envi": ["envio", "envi"],
    "despacho": ["despacho", "envio", "envi"],
    "entrega": ["entrega", "envio"],
}


def _extract_policy_stems(text: str) -> List[str]:
    """
    Extrae raíces semánticas de la query del usuario.
    Maneja typos comunes (devoluvion → devol) y conjugaciones (devolver → devol).
    """
    normalized = _normalize(text)
    found_stems: List[str] = []
    for root in _STEM_SYNONYMS:
        if root in normalized:
            found_stems.extend(_STEM_SYNONYMS[root])
    return list(dict.fromkeys(found_stems))  # dedup preservando orden


def _detect_policy_topic(query: str) -> Optional[str]:
    q = _normalize(query)

    if "cancel" in q:
        return "returns"

    if "devol" in q or "reembols" in q or ("cambi" in q and "direccion" not in q and "envio" not in q):
        return "returns"

    if any(term in q for term in ["direccion", "direccion de envio", "modificar direccion", "cambiar direccion", "antes del despacho", "despacho", "en camino", "llega", "demora", "tarda", "envio", "entrega", "seguimiento", "tracking", "guia", "transportadora", "fines de semana", "festivos", "intento de entrega", "intentos de entrega", "costo de envio", "cobran envio"]):
        return "shipping"

    # Devolución cubre: devolver, devolución, devoluvion (typo), devuelvo, reembolso, cambio
    if any(stem in q for stem in ["devol", "reembols", "cambi", "cambio"]):
        return "returns"

    # Garantía
    if any(stem in q for stem in ["garantia", "garantiz", "garant", "falla", "defecto", "reparaci", "tecnico", "instal", "instalacion"]):
        return "warranty"

    # Envío / entrega
    if any(stem in q for stem in ["envio", "envi", "entrega", "despacho", "transito", "guia", "seguimiento", "tracking", "lleg", "demor", "tard"]):
        return "shipping"

    return None


def _source_matches_topic(source: str, topic: Optional[str]) -> bool:
    if topic is None:
        return True

    s = _normalize(source)
    if topic == "returns":
        return "devol" in s or "reembolso" in s or "cambio" in s
    if topic == "warranty":
        return "garant" in s or "warrant" in s
    if topic == "shipping":
        return "env" in s or "entrega" in s or "despacho" in s
    return True


def _document_boost(query: str, source: str) -> int:
    topic = _detect_policy_topic(query)
    if _source_matches_topic(source, topic):
        return 50 if topic is not None else 0
    if topic is not None:
        return -20
    return 0


def _subtopic_header_boost(query: str, header: str) -> int:
    topic = _detect_policy_topic(query)
    subtopic = _detect_policy_subtopic(query, topic)
    if not subtopic:
        return 0

    normalized_header = _normalize(header)
    header_map = {
        "shipping": {
            "delivery_time": ["tiempos estimados", "tiempo de procesamiento", "cobertura de envios"],
            "weekend_schedule": ["tiempo de procesamiento"],
            "address_change": ["modificacion de direccion de envio", "reprogramacion de entregas"],
            "tracking": ["seguimiento de pedidos", "informacion de tracking"],
            "delivery_attempts": ["intentos de entrega"],
            "shipping_cost": ["costos de envio", "envio gratuito", "tarifas estandar"],
        },
        "returns": {
            "refund": ["plazos de reembolso", "reembolso al medio de pago", "datos para reembolso", "comprobante de reembolso"],
            "exchange": ["cambio por otro producto", "opciones de devolucion"],
            "discount_items": ["articulos no elegibles para devolucion"],
            "cancellation": ["cancelacion de pedidos"],
            "process": ["proceso de devolucion", "paso 1 solicitar la devolucion"],
            "deadline": ["plazos para devoluciones y cambios"],
        },
        "warranty": {
            "coverage": ["que cubre la garantia", "cobertura general", "vigencia de la garantia por categoria"],
            "exclusions": ["no cubierto", "exclusiones especificas por categoria"],
            "process": ["proceso de reclamacion de garantia", "si tu producto presenta fallas de funcionamiento"],
            "term": ["vigencia de la garantia por categoria"],
        },
    }

    preferred_headers = header_map.get(topic, {}).get(subtopic, [])
    for preferred in preferred_headers:
        if preferred in normalized_header:
            return 45
    return 0


def _header_boost(query: str, header: str) -> int:
    q = _normalize(query)
    h = _normalize(header)
    boost = 0

    pairs = [
        (["direccion"], ["direccion"]),
        (["reembols", "reembolso"], ["reembols", "reembolso"]),
        (["devol", "devoluci", "devolver"], ["devol", "devoluci"]),
        (["garantia", "garantiz"], ["garantia"]),
        (["envio", "envi", "entrega"], ["envio", "entrega", "despacho"]),
        (["demora", "tarda", "llega", "llegar"], ["tiempo", "entrega", "envio", "procesamiento"]),
        (["plazo", "tiempo", "cuanto"], ["plazo", "tiempo"]),
        (["cancelar", "cancelaci"], ["cancelar", "cancelaci"]),
        (["cambio", "cambi"], ["cambio", "devol"]),
    ]
    for query_stems, header_stems in pairs:
        if any(qs in q for qs in query_stems) and any(hs in h for hs in header_stems):
            boost += 15

    boost += _subtopic_header_boost(query, header)
    return boost


def _stem_score(query_stems: List[str], corpus_text: str) -> int:
    """
    Puntúa secciones basándose en presencia de raíces semánticas,
    no solo tokens exactos. Permite matchear aunque la query tenga typos
    o el documento use formas conjugadas diferentes.
    """
    if not query_stems or not corpus_text:
        return 0
    score = 0
    corpus = _normalize(corpus_text)
    for stem in query_stems:
        if stem in corpus:
            score += 3
    return score


def _score_section(query: str, section: Dict[str, Any], query_stems: List[str]) -> int:
    query_tokens = _tokenize(query)
    header_tokens = _tokenize(section["header"])
    text_tokens = _tokenize(section["text"])

    stopwords = {
        "como", "es", "la", "el", "los", "las", "de", "del", "sobre",
        "politica", "politicas", "quiero", "saber", "cual", "cuales",
        "me", "puedes", "podrias", "favor", "por", "una", "un",
        "puedo", "hacer", "pedir", "tengo", "hay", "esta",
    }
    query_tokens = [token for token in query_tokens if token not in stopwords]

    score = 0

    # Puntuación por tokens exactos
    for token in query_tokens:
        score += header_tokens.count(token) * 5
        score += text_tokens.count(token)

    # Puntuación por raíces semánticas (cubre conjugaciones y typos)
    score += _stem_score(query_stems, section["header"]) * 4
    score += _stem_score(query_stems, section["text"])

    normalized_header = _normalize(section["header"])
    normalized_text = _normalize(section["text"])
    normalized_query = _normalize(query)

    if normalized_query and normalized_query in normalized_header:
        score += 12
    if normalized_query and normalized_query in normalized_text:
        score += 8

    score += _document_boost(query, section["source"])
    score += _header_boost(query, section["header"])
    return score


def _truncate_text(text: str, max_chars: int = 700) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars].rsplit(" ", 1)[0].strip()
    return truncated + "..."


def _dedupe_sections(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped = []
    seen = set()

    for section in sections:
        key = (
            section.get("source", ""),
            _normalize(section.get("header", "")),
            _normalize(section.get("text", ""))[:250],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(section)

    return deduped


def search_policy_sections(query: str, top_k: int = 3) -> Dict[str, Any]:
    documents = _read_markdown_files()
    topic = _detect_policy_topic(query)

    # Filtrar documentos por tema detectado
    filtered_docs = [doc for doc in documents if _source_matches_topic(doc["source"], topic)]
    if filtered_docs:
        documents = filtered_docs

    # Extraer raíces semánticas de la query para scoring flexible
    query_stems = _extract_policy_stems(query)

    all_sections = []
    for doc in documents:
        all_sections.extend(_split_markdown_by_headers(doc["source"], doc["content"]))

    scored_sections = []
    for section in all_sections:
        clean_header = _clean_markdown_text(section["header"])
        clean_text = _clean_markdown_text(section["text"])

        score = _score_section(
            query,
            {
                "source": section["source"],
                "header": clean_header,
                "text": clean_text,
            },
            query_stems,
        )

        # Incluir secciones con score > 0 O que sean del documento correcto
        # Esto garantiza resultados incluso cuando la query tiene typos graves
        if score > 0 or (topic is not None and _source_matches_topic(section["source"], topic)):
            scored_sections.append({
                "source": section["source"],
                "header": clean_header,
                "text": clean_text,
                "score": score,
            })

    scored_sections.sort(key=lambda x: x["score"], reverse=True)
    scored_sections = _dedupe_sections(scored_sections)
    effective_top_k = max(top_k, 6 if topic else top_k)
    top_sections = scored_sections[:effective_top_k]

    if not top_sections:
        output = {
            "success": False,
            "message": "No encontré información relevante en las políticas para responder esa consulta.",
        }
        add_tool_trace("search_policy_sections", {"query": query, "top_k": top_k}, output)
        return output

    structured_results = []
    for section in top_sections:
        structured_results.append({
            "title": section["header"].replace("#", "").strip(),
            "content": _truncate_text(section["text"], 600),
            "source": section["source"],
            "score": section["score"],
        })

    primary_source = top_sections[0]["source"] if top_sections else None
    context_sections = []
    if primary_source:
        for section in all_sections:
            if section["source"] != primary_source:
                continue
            context_sections.append({
                "title": _clean_markdown_text(section["header"]).replace("#", "").strip(),
                "content": _clean_markdown_text(section["text"]),
                "source": section["source"],
            })

    output = {
        "success": True,
        "query": query,
        "results": structured_results,
        "context_sections": context_sections,
        "summary_hint": "Usa esta información para responder de forma clara, resumida y conversacional. No copies el texto literal, reformúlalo en lenguaje natural.",
    }

    add_tool_trace("search_policy_sections", {"query": query, "top_k": top_k}, output)
    return output


def format_policy_response(result: Dict[str, Any]) -> str:
    if not result["success"]:
        return result["message"]

    sections = result.get("context_sections") or result.get("results", [])
    if not sections:
        return "No encontré información suficiente para responder con precisión."

    query = result.get("query", "")
    topic = _detect_policy_topic(query)

    if topic == "returns":
        return _format_returns_policy_response(result, sections)
    if topic == "warranty":
        return _format_warranty_policy_response(result, sections)
    if topic == "shipping":
        return _format_shipping_policy_response(result, sections)

    formatted = []
    for sec in sections[:2]:
        title = sec.get("title", "Sección relevante").strip()
        content = _first_sentence(sec.get("content", ""))
        if title and content:
            formatted.append(f"{title}: {content}")

    if not formatted:
        return "No encontré información suficiente para responder con precisión."

    return " ".join(formatted)
