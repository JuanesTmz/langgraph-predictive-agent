"""
LangGraph — Agente ReAct con Ollama
=====================================
Agente que razona y usa herramientas (ReAct: Reason + Act).

Flujo del grafo:
  START -> agente -> ¿herramienta? -> herramientas -> agente -> ... -> END

Requisitos:
  pip install langgraph langchain-ollama langchain-core
  ollama pull qwen2.5:3b   ← modelo gratuito y liviano (~2 GB)

Modelos alternativos (cambiar MODEL en config):
  ollama pull llama3.2         (~2 GB)
  ollama pull mistral:7b       (~4 GB)
"""

import json
import math
from datetime import datetime
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
MODEL = "qwen2.5:3b"

# ---------------------------------------------
# PERSONALIDAD — mensaje de sistema que define el comportamiento del agente
# ---------------------------------------------
SYSTEM_PROMPT = """Eres Matix, un asistente inteligente, directo y con buen humor.
Tu personalidad:
- Eres preciso y vas al grano, sin rodeos innecesarios.
- Usas un tono amigable y cercano, como un colega que sabe mucho.
- Cuando algo te divierte, lo dices. Cuando algo no tiene sentido, lo aclaras con respeto.
- Nunca inventas resultados matemáticos — para eso tienes herramientas.
- Siempre explicas brevemente qué hiciste y por qué, en una sola oración."""


# ---------------------------------------------
# 1. STATE — lista de mensajes con reducción automática (add_messages)
# ---------------------------------------------
class State(TypedDict):
    # Annotated[list, add_messages] fusiona listas en lugar de reemplazarlas
    messages: Annotated[list, add_messages]


# ---------------------------------------------
# 2. TOOLS — herramientas que el agente puede invocar
# ---------------------------------------------
@tool
def calculadora(expresion: str) -> str:
    """
    Evalúa una expresión matemática y retorna el resultado.
    Usa esta herramienta cuando el usuario pida calcular, operar o resolver
    cualquier expresión numérica. Nunca calcules mentalmente — siempre delega
    aquí para garantizar precisión.
    Ejemplos: '2 + 2', '10 * 5', 'sqrt(16)', '2 ** 8'
    """
    try:
        # Entorno seguro: solo funciones de math y operaciones básicas
        entorno = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        return str(eval(expresion, {"__builtins__": {}}, entorno))  # noqa: S307
    except Exception as e:
        return f"Error evaluando '{expresion}': {e}"


TOOLS = [calculadora]
TOOL_MAP = {t.name: t for t in TOOLS}


# ---------------------------------------------
# 3. LLM con herramientas enlazadas
# ---------------------------------------------
llm = ChatOllama(model=MODEL)
llm_con_tools = llm.bind_tools(TOOLS)


# ---------------------------------------------
# 4. NODES
# ---------------------------------------------
def nodo_agente(state: State) -> dict:
    """Llama al LLM y decide si usar una herramienta o responder directo."""
    mensajes_con_sistema = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    respuesta = llm_con_tools.invoke(mensajes_con_sistema)
    return {"messages": [respuesta]}


def nodo_herramientas(state: State) -> dict:
    """Ejecuta las herramientas que el LLM solicitó."""
    ultimo_mensaje: AIMessage = state["messages"][-1]
    resultados = []

    for tool_call in ultimo_mensaje.tool_calls:
        nombre = tool_call["name"]
        args = tool_call["args"]
        herramienta = TOOL_MAP.get(nombre)

        if herramienta:
            try:
                salida = herramienta.invoke(args)
            except Exception as e:
                salida = f"Error al ejecutar '{nombre}': {e}. Revisa los argumentos e intenta de nuevo."
        else:
            salida = f"Herramienta '{nombre}' no encontrada."

        resultados.append(
            ToolMessage(
                content=str(salida),
                tool_call_id=tool_call["id"],
                name=nombre,
            )
        )

    return {"messages": resultados}


# ---------------------------------------------
# 5. ROUTING — ¿el agente quiere usar una herramienta?
# ---------------------------------------------
def debe_continuar(state: State) -> str:
    ultimo = state["messages"][-1]
    if isinstance(ultimo, AIMessage) and ultimo.tool_calls:
        return "herramientas"
    return END


# ---------------------------------------------
# 6. GRAFO
# ---------------------------------------------
def construir_agente():
    grafo = StateGraph(State)

    grafo.add_node("agente", nodo_agente)
    grafo.add_node("herramientas", nodo_herramientas)

    grafo.add_edge(START, "agente")
    grafo.add_conditional_edges(
        "agente",
        debe_continuar,
        {"herramientas": "herramientas", END: END},
    )
    grafo.add_edge("herramientas", "agente")  # siempre vuelve al agente tras usar tool

    return grafo.compile()


# ---------------------------------------------
# 7. CHAT INTERACTIVO
# ---------------------------------------------
def chat_interactivo(app):
    """Loop de conversacion por consola con historial y reporte de tools."""
    mensajes = []

    print("\nAgente listo. Escribe 'salir' para terminar.\n")
    print(f"Tools disponibles: {', '.join(TOOL_MAP.keys())}\n")
    print("-" * 55)

    while True:
        try:
            entrada = input("Tu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if not entrada:
            continue
        if entrada.lower() in ("salir", "exit", "quit"):
            print("Hasta luego.")
            break

        mensajes.append(HumanMessage(content=entrada))

        tools_usadas = []
        estado_final = None

        for evento in app.stream({"messages": mensajes}, stream_mode="values"):
            estado_final = evento
            ultimo = evento["messages"][-1]
            if isinstance(ultimo, ToolMessage):
                tools_usadas.append(ultimo.name)

        if estado_final is None:
            print("Agente: [sin respuesta]")
            continue

        mensajes = estado_final["messages"]  # mantiene historial completo
        respuesta = mensajes[-1].content

        print(f"\nAgente: {respuesta}")

        if tools_usadas:
            print(f"[Tools usadas: {', '.join(tools_usadas)}]")
        else:
            print("[Sin tools — respuesta directa del modelo]")

        print("-" * 55)

    if mensajes:
        nombre_archivo = f"sesion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        estado_serializado = [
            {"tipo": type(m).__name__, "contenido": m.content}
            for m in mensajes
        ]
        with open(nombre_archivo, "w", encoding="utf-8") as f:
            json.dump(estado_serializado, f, ensure_ascii=False, indent=2)
        print(f"Estado de sesión guardado en: {nombre_archivo}")


# ---------------------------------------------
# 8. MAIN
# ---------------------------------------------
if __name__ == "__main__":
    print(f"Iniciando agente con modelo: {MODEL}")
    print("Asegurate de que Ollama este corriendo -> 'ollama serve'")

    app = construir_agente()
    chat_interactivo(app)
