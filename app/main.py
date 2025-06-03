from fastapi import FastAPI
from pydantic import BaseModel
from . import rag_chain

app = FastAPI()

class Question(BaseModel):
    query: str
    chat_history: list = []

@app.post("/ask")
async def ask_question(question: Question):
    result = rag_chain.retrieve_and_generate(
        query=question.query,
        chat_history=question.chat_history
    )
    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }