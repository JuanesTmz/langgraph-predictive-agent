"""
Ensamblaje del grafo multiagente con LangGraph.

Topología:
                    ┌──────────────┐
         START ────▶│ router_agent │
                    └──────┬───────┘
                           │
              ─────────────┴──────────────
              │  conditional_edge        │
              │  (route_after_router)    │
              ▼                          ▼
        ┌───────────┐           ┌──────────────────┐
        │ rag_agent │           │clarification_agent│
        └─────┬─────┘           └────────┬─────────┘
              │                          │
              ▼                          │
    ┌───────────────────┐                │
    │ synthesizer_agent │                │
    └─────────┬─────────┘                │
              │                          │
              └────────────┬─────────────┘
                           ▼
                          END

Conceptos de LangGraph que se ilustran aquí:
  - StateGraph       : grafo cuyo estado es un TypedDict compartido
  - add_node         : registra una función Python como nodo ejecutable
  - add_edge         : arco fijo (A siempre va a B)
  - add_conditional_edges : arco condicional (decide el siguiente nodo en tiempo de ejecución)
  - compile          : convierte el grafo en un Runnable invocable
"""

from functools import partial

from langgraph.graph import END, START, StateGraph

from agents import clarification_agent, rag_agent, router_agent, synthesizer_agent
from rag import build_vectorstores
from state import AgentState


# ── Función de enrutamiento condicional ───────────────────────────────────────

def route_after_router(state: AgentState) -> str:
    """
    Decide cuál es el siguiente nodo después del Router Agent.

    LangGraph llama a esta función con el estado actual y usa el string
    retornado como clave para elegir el nodo de destino en el mapa
    definido en `add_conditional_edges`.
    """
    if state["department"] == "Desconocido":
        return "clarification"
    return "rag"


# ── Construcción del grafo ─────────────────────────────────────────────────────

def build_graph():
    """
    Construye y compila el StateGraph completo.

    El vectorstore se construye UNA sola vez aquí y se inyecta en el
    rag_agent mediante functools.partial, para no reconstruirlo en cada
    consulta (los embeddings son costosos de calcular).
    """
    print("Cargando documentos y construyendo índices vectoriales...")
    vectorstores = build_vectorstores()

    # StateGraph tipado: LangGraph valida que los nodos devuelvan claves
    # que existan en AgentState.
    workflow = StateGraph(AgentState)

    # ── Registro de nodos ──────────────────────────────────────────────────────
    # Cada nodo es una función (state) -> dict con las claves a actualizar.
    # partial() inyecta los vectorstores en rag_agent sin romper la firma.
    workflow.add_node("router",        router_agent)
    workflow.add_node("rag",           partial(rag_agent, vectorstores=vectorstores))
    workflow.add_node("synthesizer",   synthesizer_agent)
    workflow.add_node("clarification", clarification_agent)

    # ── Arcos del grafo ────────────────────────────────────────────────────────
    # Arco fijo: siempre empieza en el router
    workflow.add_edge(START, "router")

    # Arco condicional: el router decide si hay suficiente información para RAG
    # o si hay que pedirle más detalles al usuario.
    workflow.add_conditional_edges(
        source="router",
        path=route_after_router,
        path_map={"rag": "rag", "clarification": "clarification"},
    )

    # Arcos fijos del camino feliz: RAG → Synthesizer → fin
    workflow.add_edge("rag",           "synthesizer")
    workflow.add_edge("synthesizer",   END)
    workflow.add_edge("clarification", END)

    # compile() valida la estructura del grafo y devuelve un Runnable
    return workflow.compile()
