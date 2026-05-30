from typing import TypedDict


class AgentState(TypedDict):
    """
    Estado compartido que fluye entre todos los nodos del grafo.

    query             : pregunta original del empleado
    entities          : entidades nombradas detectadas por spaCy (NUEVO)
    keywords          : términos clave extraídos por spaCy (NUEVO)
    sentiment         : sentimiento detectado por Transformers: positive | neutral | negative (NUEVO)
    urgency_level     : nivel de urgencia derivado del sentimiento: high | medium | low (NUEVO)
    department        : departamento detectado por el Router Agent
    retrieved_context : fragmentos recuperados por el RAG Agent
    response          : respuesta final generada para el empleado
    """
    query: str
    entities: list
    keywords: list
    sentiment: str
    urgency_level: str
    department: str
    retrieved_context: str
    response: str
