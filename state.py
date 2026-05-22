from typing import TypedDict


class AgentState(TypedDict):
    """
    Estado compartido que fluye entre todos los nodos del grafo.

    query             : pregunta original del empleado
    department        : departamento detectado por el Router Agent
    retrieved_context : fragmentos recuperados por el RAG Agent
    response          : respuesta final generada para el empleado
    """
    query: str
    department: str
    retrieved_context: str
    response: str
