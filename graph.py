"""
Ensamblaje del grafo multiagente con LangGraph.

         START
           │
    ┌──────▼────────┐
    │  preprocess   │  spaCy — extrae entidades y palabras clave
    └──────┬────────┘
           │
    ┌──────▼────────┐
    │   sentiment   │  HuggingFace Transformers — detecta frustración/urgencia
    └──────┬────────┘
           │
    ┌──────▼────────┐
    │    router     │  LLM — clasifica en IT / RRHH / Vacaciones / Incapacidades
    └──────┬────────┘
           │
    ───────┴───────────────────
    │                         │
    ▼                         ▼
┌───────┐             ┌───────────────┐
│  rag  │             │ clarification │
└───┬───┘             └───────┬───────┘
    │                         │
    ▼                         │
┌─────────────┐               │
│ synthesizer │               │
└──────┬──────┘               │
       └──────────┬───────────┘
                  ▼
                 END
"""

from functools import partial

from langgraph.graph import END, START, StateGraph

from agents.llm_agents import clarification_agent, router_agent, synthesizer_agent
from agents.nlp_agents import preprocess_agent, sentiment_agent
from agents.rag import build_vectorstores, rag_agent
from state import AgentState


def route_after_router(state: AgentState) -> str:
    return "clarification" if state["department"] == "Desconocido" else "rag"


def build_graph():
    print("Cargando documentos y construyendo vectorstores...")
    vectorstores = build_vectorstores()

    workflow = StateGraph(AgentState)

    workflow.add_node("preprocess",    preprocess_agent)
    workflow.add_node("sentiment",     sentiment_agent)
    workflow.add_node("router",        router_agent)
    workflow.add_node("rag",           partial(rag_agent, vectorstores=vectorstores))
    workflow.add_node("synthesizer",   synthesizer_agent)
    workflow.add_node("clarification", clarification_agent)

    workflow.add_edge(START,         "preprocess")
    workflow.add_edge("preprocess",  "sentiment")
    workflow.add_edge("sentiment",   "router")

    workflow.add_conditional_edges(
        source="router",
        path=route_after_router,
        path_map={"rag": "rag", "clarification": "clarification"},
    )

    workflow.add_edge("rag",           "synthesizer")
    workflow.add_edge("synthesizer",   END)
    workflow.add_edge("clarification", END)

    return workflow.compile()
