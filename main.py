"""
Punto de entrada del Asistente de Soporte Empresarial Multiagente.

Uso:
    python main.py

El sistema inicializa los modelos NLP al arrancar (una sola vez) y luego
queda en un loop interactivo esperando preguntas del empleado.

Pipeline de agentes por consulta:
  1. PreprocessAgent  (spaCy)                 → entidades + keywords
  2. SentimentAgent   (HuggingFace Transformers) → sentimiento + urgencia
  3. RouterAgent      (LLM qwen2.5:3b)          → departamento
  4a. RagAgent        (sentence-transformers)   → contexto recuperado
  4b. ClarificationAgent (estático)            → [si no reconoce departamento]
  5. SynthesizerAgent (LLM qwen2.5:3b)         → respuesta final
"""

import os
import sys
import warnings

# UTF-8 en stdout (Windows cp1252 por defecto)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Silenciar warnings de HuggingFace Hub (caché de symlinks en Windows)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message=".*symlinks.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

from graph import build_graph

_URGENCY_LABEL = {"high": "ALTA", "medium": "MEDIA", "low": "baja"}
_SENTIMENT_LABEL = {"positive": "positivo", "negative": "negativo", "neutral": "neutral"}


def main() -> None:
    graph = build_graph()

    print("\n" + "=" * 65)
    print("  ASISTENTE DE SOPORTE EMPRESARIAL  —  Powered by LangGraph")
    print("  NLP: spaCy + HuggingFace Transformers + sentence-transformers")
    print("=" * 65)
    print("Areas disponibles: IT | RRHH | Vacaciones | Incapacidades")
    print("Escribe 'salir' para terminar.")
    print("=" * 65)

    while True:
        query = input("\nEmpleado > ").strip()

        if not query:
            continue
        if query.lower() in ("salir", "exit", "quit"):
            print("\nHasta luego. Que tengas un excelente dia!")
            break

        initial_state = {
            "query":             query,
            "entities":          [],
            "keywords":          [],
            "sentiment":         "neutral",
            "urgency_level":     "low",
            "department":        "",
            "retrieved_context": "",
            "response":          "",
        }

        result = graph.invoke(initial_state)

        # ── Análisis NLP ───────────────────────────────────────────────────────
        entities  = result.get("entities", [])
        keywords  = result.get("keywords", [])
        sentiment = result.get("sentiment", "neutral")
        urgency   = result.get("urgency_level", "low")

        print("\n" + "-" * 65)
        print("[spaCy]")
        print(f"  Entidades : {', '.join(entities) if entities else '—'}")
        print(f"  Palabras clave: {', '.join(keywords) if keywords else '—'}")
        print(f"[Transformers] Sentimiento: {_SENTIMENT_LABEL.get(sentiment, sentiment)}"
              f"  |  Urgencia: {_URGENCY_LABEL.get(urgency, urgency)}")
        print(f"[Router LLM]  Departamento: {result['department']}")
        print("-" * 65)
        print(f"\nAsistente > {result['response']}")
        print("\n" + "-" * 65)


if __name__ == "__main__":
    main()
