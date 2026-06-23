# Agente Conversacional Autónomo – Challenge Strata Analytics

## Descripción General

Este proyecto implementa un agente conversacional autónomo para un entorno de e-commerce, capaz de responder consultas de clientes combinando:

* Datos estructurados (pedidos, clientes, productos)
* Documentos de políticas en Markdown
* Modelos de IA generativa (OpenAI / Amazon Bedrock)

El agente está diseñado para operar bajo principios de:

* seguridad
* trazabilidad
* no alucinación
* eficiencia operativa

---

## Arquitectura del Agente

El sistema sigue una arquitectura modular:

* `core/agent.py` → Orquestador principal del agente
* `core/router.py` → Clasificación de intención (routing)
* `core/tools.py` → Acceso a datos estructurados
* `core/policy_rag.py` → Recuperación de políticas (RAG)
* `core/guards.py` → Seguridad y control de acceso
* `core/response_validator.py` → Validación anti-alucinación
* `core/session_context.py` → Trazabilidad y estado de sesión
* `core/llm_client.py` → Abstracción de proveedor LLM (OpenAI / Bedrock)

---

## Lógica de Enrutamiento

El agente distingue entre:

1. **FAQ públicas**

   * Métodos de pago, cobertura, canales
   * No requieren autenticación

2. **Consultas de políticas**

   * Resueltas exclusivamente mediante RAG
   * No se usa conocimiento interno del modelo

3. **Consultas de catálogo**

   * Precio, stock general
   * No requieren autenticación

4. **Consultas de pedidos (sensibles)**

   * Estado, historial, devoluciones, montos
   * Requieren autenticación obligatoria

---

## Seguridad y Control de Acceso

El agente implementa un **gate obligatorio**:

* Verificación por:

  * documento (`dni`)
  * teléfono (`phone`)
* Sin autenticación:

  * no se permite acceso a datos de pedidos

### Protección contra prompt injection

Se detectan intentos de manipulación como:

* “ignora tus instrucciones”
* “soy administrador”
* “omite validación”

Estos son bloqueados antes del flujo principal.

---

## Regla Anti-Alucinación

El agente **nunca responde datos sin evidencia**.

Para cada consulta sensible:

* Se identifica el intent
* Se ejecuta la herramienta correspondiente
* Se valida que la herramienta fue usada en ese turno
* Se bloquea la respuesta si no hay trazabilidad

Esto se implementa mediante:

* `add_tool_trace(...)`
* validación con `get_tool_trace_since(...)`

---

## Recuperación Documental (RAG)

Las políticas se procesan por secciones (Markdown):

* Se realiza ranking por relevancia
* Se selecciona solo el contenido necesario
* No se inyectan documentos completos al modelo

El agente responde **únicamente con base en el contenido recuperado**.

---

## Soporte de IA Generativa

El agente es **multiproveedor**:

* OpenAI → desarrollo local
* Amazon Bedrock → entorno de evaluación

### Configuración recomendada para evaluación

* Provider: Bedrock
* Modo estricto activado

Esto garantiza que:

* no haya fallback a lógica legacy
* todas las respuestas dependan de IA generativa + tools

---

## Observabilidad del Agente

Todas las acciones relevantes son registradas:

* uso de herramientas
* inputs y outputs
* identidad del cliente

Funciones clave:

* `add_tool_trace()`
* `set_session_customer()`
* `get_tool_trace()`
* `get_tool_trace_since()`

Esto permite auditar completamente el comportamiento del agente.

---

## Eficiencia y Optimización

El agente reduce el uso del LLM mediante:

* routing previo
* uso de herramientas cuando es posible
* respuestas determinísticas para FAQ

Esto permite:

* menor consumo de tokens
* menor latencia
* mejor uso del presupuesto en AWS

---

## ⏱Tiempo de Respuesta (TTFT)

El diseño evita llamadas innecesarias al modelo y mantiene respuestas dentro del límite requerido (< 10s).

---

##  Casos de Uso Cubiertos

* Consulta de estado de pedidos
* Validación de devoluciones
* Consulta de políticas
* Información de productos
* Manejo de autenticación
* Bloqueo de accesos no autorizados

---

##  Ejecución

El agente se instancia mediante:

```python
agent = create_agent()
response = agent("texto del usuario")
```

El objeto retornado:

* es invocable
* retorna texto válido
* nunca es `None`

---

## Consideraciones Finales

Este agente fue diseñado siguiendo los requerimientos del challenge:

* Cumple con el gate de autenticación
* Implementa trazabilidad obligatoria
* Aplica la regla anti-alucinación
* Integra RAG con documentos
* Soporta ejecución en Amazon Bedrock

El enfoque principal fue garantizar:

* precisión
* seguridad
* y comportamiento auditable

---
