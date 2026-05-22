"""
Punto de entrada del Asistente de Soporte Empresarial Multiagente.

Uso:
    python main.py

El sistema construye los índices RAG al inicio (una sola vez) y luego
queda en un loop interactivo esperando preguntas del empleado.
"""

import sys

# Forzar UTF-8 en la salida estándar (necesario en Windows con cp1252)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from graph import build_graph


def main() -> None:
    graph = build_graph()

    print("\n" + "=" * 65)
    print("  ASISTENTE DE SOPORTE EMPRESARIAL  —  Powered by LangGraph")
    print("=" * 65)
    print("Áreas disponibles: IT | RRHH | Vacaciones | Incapacidades")
    print("Escribe 'salir' para terminar.")
    print("=" * 65)

    while True:
        query = input("\nEmpleado > ").strip()

        if not query:
            continue
        if query.lower() in ("salir", "exit", "quit"):
            print("\nHasta luego. ¡Que tengas un excelente día!")
            break

        # Estado inicial: solo la query; el resto lo completan los agentes
        initial_state = {
            "query": query,
            "department": "",
            "retrieved_context": "",
            "response": "",
        }

        result = graph.invoke(initial_state)

        print(f"\n[Departamento detectado: {result['department']}]")
        print(f"\nAsistente > {result['response']}")
        print("\n" + "-" * 65)


if __name__ == "__main__":
    main()
