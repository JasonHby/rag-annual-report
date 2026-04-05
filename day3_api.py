import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from day2_retrieval import search_database

# define ask request and response models
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list[dict]

# initiate FastAPI app
app = FastAPI(title = "Matinrea RAG API")

# initiate LLM
#    这里虽然使用的是 ChatOpenAI 封装器，
#    但实际调用的是 Groq 提供的 OpenAI-compatible API。
llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an internal company assistant.

Answer the user's question only based on the context below.
If the answer cannot be found in the context, say "I don't know based on the provided context."

When possible:
1. Use the exact figure from the context.
2. Mention the source page briefly.
3. Keep the answer concise and factual.

Context:
{context}

Question:
{question}

Answer:
"""
)


def format_context(docs_with_scores):
    context_parts = []
    for i, (doc, score) in enumerate(docs_with_scores, start = 1):
        source = doc.metadata.get("source", "unknown")
        page_label = doc.metadata.get("page_label", "unknown")

        context_parts.append(
            f"[Source {i} | file = {source} | page = {page_label} | score = {score}]\n"
            f"{doc.page_content}\n"
        )
    return "\n\n".join(context_parts)

def format_sources(docs_with_scores):
    sources = []
    for doc, score in docs_with_scores:
        sources.append(
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "unknown"),
                "page_label": doc.metadata.get("page_label", "unknown"),
                "score": score,
                "content": doc.page_content,
            }
        )
    return sources

@app.get("/")
def root():
    return {"message": "Matinrea RAG API is running. Use the /api/chat endpoint to ask questions."}

@app.post("/api/chat", response_model = AskResponse)
def chat(request: AskRequest):
    docs_with_scores = search_database(request.question, top_k=3)
    context = format_context(docs_with_scores)

    prompt = prompt_template.format(
        context = context,
        question = request.question
    )

    response = llm.invoke(prompt)
    return AskResponse(
        answer = response.content,
        sources = format_sources(docs_with_scores)
    )
