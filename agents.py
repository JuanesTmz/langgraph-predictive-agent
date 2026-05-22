"""
Definición de los 4 nodos agentes del grafo multiagente.

Arquitectura:
  ┌─────────────┐     ┌─────────────┐     ┌───────────────────┐
  │ RouterAgent │────▶│   RagAgent  │────▶│ SynthesizerAgent  │
  └─────────────┘     └─────────────┘     └───────────────────┘
         │
         │ (si departamento = "Desconocido")
         ▼
  ┌──────────────────┐
  │ ClarificationAgent│
  └──────────────────┘

Todos los agentes usan el mismo modelo local qwen2.5:3b vía Ollama.
Cada uno tiene un rol y un system prompt distinto.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from rag import retrieve
from state import AgentState

# Un único modelo compartido por los 3 agentes LLM.
# temperature=0 para respuestas deterministas y clasificaciones consistentes.
_LLM = ChatOllama(model="qwen2.5:3b", temperature=0)

# ── Agente 1: Router ──────────────────────────────────────────────────────────

def router_agent(state: AgentState) -> dict:
    """
    NODO 1 — Router Agent.

    Responsabilidad: clasificar la pregunta del empleado en uno de los
    4 departamentos disponibles. Escribe `department` en el estado.

    Por qué un agente dedicado para esto:
      - Separa la lógica de enrutamiento de la de recuperación y síntesis.
      - Permite escalar fácilmente agregando nuevos departamentos sin tocar
        los otros agentes.
    """
    system = """Eres un clasificador de consultas para el sistema de soporte empresarial.
Tu única tarea es leer la pregunta de un empleado y responder con UNA SOLA PALABRA
que indique el departamento al que pertenece la consulta.

Palabras válidas (responde exactamente una de estas):
- IT          → problemas técnicos, computadores, software, VPN, correo, contraseñas, hardware, internet
- RRHH        → recursos humanos, beneficios, código de conducta, nómina, evaluación de desempeño, onboarding
- Vacaciones  → días libres, permisos, ausencias programadas, solicitud de vacaciones, días acumulados
- Incapacidades → enfermedad, accidente, licencia médica, incapacidad, maternidad, paternidad, luto
- Desconocido → si la consulta no encaja claramente en ninguna categoría anterior

Responde ÚNICAMENTE con la palabra. Sin explicaciones, sin puntuación."""

    response = _LLM.invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["query"]),
    ])

    # Tomar la primera palabra para tolerar respuestas con espacios extra
    department = response.content.strip().split()[0]

    valid = {"IT", "RRHH", "Vacaciones", "Incapacidades"}
    if department not in valid:
        department = "Desconocido"

    return {"department": department}


# ── Agente 2: RAG ─────────────────────────────────────────────────────────────

def rag_agent(state: AgentState, vectorstores: dict) -> dict:
    """
    NODO 2 — RAG Agent.

    Responsabilidad: dado el departamento detectado, buscar en el vectorstore
    correspondiente los fragmentos de documento más relevantes para la pregunta.
    Escribe `retrieved_context` en el estado.

    Aquí ocurre el núcleo de RAG:
      1. La pregunta se convierte en un vector (embedding).
      2. Se calcula la similitud coseno contra todos los chunks indexados.
      3. Se devuelven los k chunks más similares semánticamente.
    """
    context = retrieve(
        query=state["query"],
        department=state["department"],
        vectorstores=vectorstores,
        k=3,
    )
    return {"retrieved_context": context}


# ── Agente 3: Synthesizer ─────────────────────────────────────────────────────

def synthesizer_agent(state: AgentState) -> dict:
    """
    NODO 3 — Synthesizer Agent.

    Responsabilidad: generar una respuesta profesional, empática y citada
    a partir del contexto recuperado. Escribe `response` en el estado.

    La restricción de usar SOLO el contexto recuperado es intencional:
    evita que el LLM "alucine" respuestas que no están en los documentos
    internos de la empresa.
    """
    system = """Eres el asistente de soporte interno de la empresa. Eres profesional, empático y claro.

Reglas estrictas:
1. Responde EXCLUSIVAMENTE con información del contexto proporcionado.
2. Si el contexto no contiene la respuesta, dilo explícitamente: "Esta información no está disponible en nuestra base de conocimiento interna."
3. Cita los fragmentos que usaste con [1], [2] o [3] al final de cada oración relevante.
4. Usa un tono cercano pero profesional.
5. Si hay pasos a seguir, enuméralos claramente.
6. Responde en español."""

    prompt = f"""Contexto recuperado de la base de conocimiento interna:
{state["retrieved_context"]}

Pregunta del empleado:
{state["query"]}"""

    response = _LLM.invoke([
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ])

    return {"response": response.content}


# ── Agente 4: Clarification (nodo fallback) ───────────────────────────────────

def clarification_agent(state: AgentState) -> dict:
    """
    NODO 4 — Clarification Agent.

    Responsabilidad: responder cuando el Router no pudo clasificar la consulta.
    No usa LLM — respuesta estática y determinista, sin riesgo de alucinación.

    Este nodo demuestra que en LangGraph no todos los nodos tienen que ser LLMs:
    pueden ser funciones Python simples o llamadas a APIs externas.
    """
    response = (
        "No pude identificar el área de tu consulta. "
        "Por favor, reformúlala indicando si es sobre:\n\n"
        "- **IT**: problemas técnicos, software, VPN, correo electrónico, contraseñas\n"
        "- **RRHH**: recursos humanos, beneficios, conducta, nómina, evaluación de desempeño\n"
        "- **Vacaciones**: solicitud o consulta sobre días de vacaciones o permisos\n"
        "- **Incapacidades**: licencias médicas, accidentes laborales o incapacidades\n\n"
        "También puedes escribir directamente al área correspondiente: rrhh@empresa.com"
    )
    return {"response": response}
