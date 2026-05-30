"""
Agentes basados en LLM: Router, Synthesizer y Clarification.

  router_agent       →  clasifica la pregunta en un departamento
  synthesizer_agent  →  redacta la respuesta usando el contexto recuperado
  clarification_agent→  respuesta estática cuando no se reconoce el departamento

Los tres usan el modelo local qwen2.5:3b vía Ollama (temperature=0 para
respuestas deterministas y reproducibles).
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from .rag import retrieve

_LLM = ChatOllama(model="qwen2.5:3b", temperature=0)


def router_agent(state: dict) -> dict:
    """
    NODO — Router Agent (LLM).

    Recibe la pregunta del empleado y responde con UNA SOLA PALABRA:
    IT | RRHH | Vacaciones | Incapacidades | Desconocido.

    Esa palabra determina el siguiente nodo del grafo (conditional_edge).
    """
    system = """Eres un clasificador de consultas para el sistema de soporte empresarial.
Tu única tarea es leer la pregunta de un empleado y responder con UNA SOLA PALABRA
que indique el departamento al que pertenece la consulta.

Palabras válidas (responde exactamente una de estas):
- IT          → problemas técnicos, computadores, software, VPN, correo, contraseñas, hardware
- RRHH        → recursos humanos, beneficios, conducta, nómina, evaluación, onboarding
- Vacaciones  → días libres, permisos, ausencias, solicitud de vacaciones
- Incapacidades → enfermedad, accidente, licencia médica, maternidad, paternidad
- Desconocido → si la consulta no encaja claramente en ninguna categoría anterior

Responde ÚNICAMENTE con la palabra. Sin explicaciones, sin puntuación."""

    response = _LLM.invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["query"]),
    ])

    department = response.content.strip().split()[0]
    if department not in {"IT", "RRHH", "Vacaciones", "Incapacidades"}:
        department = "Desconocido"

    return {"department": department}


def synthesizer_agent(state: dict) -> dict:
    """
    NODO — Synthesizer Agent (LLM + sentiment-aware).

    Redacta una respuesta profesional y citada usando SOLO el contexto recuperado.
    Ajusta el tono según el sentimiento detectado por el agente Transformer:
      - urgencia high   → muy empático, ofrece escalamiento
      - urgencia medium → empático y tranquilizador
      - urgencia low    → estándar

    La restricción de usar solo el contexto evita que el LLM invente información.
    """
    sentiment = state.get("sentiment", "neutral")
    urgency   = state.get("urgency_level", "low")
    entities  = state.get("entities", [])
    keywords  = state.get("keywords", [])

    if urgency == "high":
        tone_note = (
            "\nATENCION: El empleado muestra frustración alta. "
            "Sé especialmente empático, valida su experiencia y ofrece escalar el caso con un supervisor."
        )
    elif urgency == "medium":
        tone_note = "\nNota: El empleado puede estar preocupado. Responde con empatía y tranquilidad."
    else:
        tone_note = ""

    system = f"""Eres el asistente de soporte interno de la empresa. Eres profesional, empático y claro.

Análisis previo de la consulta:
- Entidades (spaCy): {', '.join(entities) if entities else 'ninguna'}
- Palabras clave (spaCy): {', '.join(keywords) if keywords else 'ninguna'}
- Sentimiento (Transformers): {sentiment}{tone_note}

Reglas:
1. Responde SOLO con información del contexto proporcionado.
2. Si el contexto no contiene la respuesta, dilo explícitamente.
3. Cita los fragmentos usados con [1], [2] o [3] al final de cada oración.
4. Si hay pasos a seguir, enuméralos claramente.
5. Responde en español."""

    prompt = f"""Contexto de la base de conocimiento:
{state["retrieved_context"]}

Pregunta del empleado:
{state["query"]}"""

    response = _LLM.invoke([
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ])

    return {"response": response.content}


def clarification_agent(state: dict) -> dict:
    """
    NODO — Clarification Agent (sin LLM, respuesta estática).

    Se activa cuando el Router no pudo clasificar la consulta.
    No usa LLM — respuesta determinista sin riesgo de alucinación.
    Demuestra que en LangGraph los nodos no tienen que ser LLMs.
    """
    return {"response": (
        "No pude identificar el área de tu consulta. "
        "Por favor, reformúlala indicando si es sobre:\n\n"
        "- **IT**: problemas técnicos, software, VPN, correo, contraseñas\n"
        "- **RRHH**: recursos humanos, beneficios, nómina, evaluación\n"
        "- **Vacaciones**: solicitud o consulta sobre días de vacaciones\n"
        "- **Incapacidades**: licencias médicas o accidentes laborales\n\n"
        "También puedes escribir a: rrhh@empresa.com"
    )}
