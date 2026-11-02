"""LangGraph RAG pipeline: retrieve relevant PDF chunks, then generate an answer."""
from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from .models import get_chat_model, get_vector_store

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using only the provided "
    "context from the user's PDF documents. If the context does not contain "
    "the answer, say you don't know instead of guessing. Cite the source "
    "file name(s) when possible."
)


class RAGState(TypedDict):
    question: str
    documents: List[Document]
    answer: str


def build_graph(k: int = 4):
    vector_store = get_vector_store()
    chat_model = get_chat_model()

    def retrieve(state: RAGState) -> RAGState:
        docs = vector_store.similarity_search(state["question"], k=k)
        return {"documents": docs}

    def generate(state: RAGState) -> RAGState:
        context = "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}, "
            f"page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in state["documents"]
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion: {state['question']}"
            ),
        ]
        response = chat_model.invoke(messages)
        return {"answer": response.content}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
