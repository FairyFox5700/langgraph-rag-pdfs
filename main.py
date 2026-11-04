"""CLI entrypoint: `python main.py ingest` or `python main.py chat`."""
import sys

from src.graph import build_graph
from src.ingest import ingest


def chat() -> None:
    app = build_graph()
    print("RAG chat ready. Type a question (or 'exit' to quit).\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        result = app.invoke({"question": question})
        print(f"\nAssistant: {result['answer']}\n")
        sources = {
            doc.metadata.get("source", "unknown") for doc in result.get("documents", [])
        }
        if sources:
            print(f"Sources: {', '.join(sorted(sources))}\n")


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "chat"
    if command == "ingest":
        ingest()
    elif command == "chat":
        chat()
    else:
        print(f"Unknown command '{command}'. Use 'ingest' or 'chat'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
