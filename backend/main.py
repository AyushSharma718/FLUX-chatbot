import os
from typing import Optional

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

HF_MODEL = os.getenv(
    "HF_MODEL",
    "deepseek-ai/DeepSeek-V3-0324"
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "chroma_db"
)

COLLECTION_NAME = "flux_website"


# ============================================================
# VALIDATE ENVIRONMENT
# ============================================================

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN not found.\n"
        "Please add HF_TOKEN to backend/.env"
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="FLUX AI Chatbot",
    description=(
        "AI-powered RAG chatbot for "
        "Future Leaders of Unbound Xperiments"
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("=" * 60)
print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print(
    f"Embedding model loaded: {EMBEDDING_MODEL}"
)


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

print("Connecting to ChromaDB...")

try:

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

except Exception as e:

    raise RuntimeError(
        "\nCould not load ChromaDB collection.\n"
        f"Collection: {COLLECTION_NAME}\n"
        f"Path: {CHROMA_PATH}\n"
        f"Error: {e}\n\n"
        "Run build_knowledge.py first."
    )


print(
    f"ChromaDB connected successfully."
)

print(
    f"Documents stored: {collection.count()}"
)


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

print("Initializing Hugging Face...")

# IMPORTANT:
# provider="auto" allows Hugging Face to select
# an available inference provider for the model.

llm_client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN
)

print("Hugging Face client initialized.")

print(
    f"LLM model: {HF_MODEL}"
)

print("=" * 60)


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

class ChatRequest(BaseModel):

    session_id: str = "default"

    message: str


class ChatResponse(BaseModel):

    reply: str


# ============================================================
# SEARCH WEBSITE
# ============================================================

def search_website(
    question: str,
    n_results: int = 4
):
    """
    Convert the user's question into an embedding
    and search the FLUX website knowledge base.
    """

    # Create embedding for user question
    question_embedding = embedding_model.encode(
        question
    ).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[
            question_embedding
        ],
        n_results=n_results
    )

    return results


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results) -> str:
    """
    Convert ChromaDB search results into context
    that can be given to the LLM.
    """

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    if not documents:

        return (
            "No relevant information was found "
            "in the FLUX website knowledge base."
        )

    context_parts = []

    for index, document in enumerate(documents):

        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        )

        title = metadata.get(
            "title",
            "FLUX Website"
        )

        url = metadata.get(
            "url",
            ""
        )

        context_parts.append(
            f"""
SOURCE {index + 1}

Title:
{title}

URL:
{url}

Website Content:
{document}

--------------------------------------------------
"""
        )

    return "\n".join(context_parts)


# ============================================================
# GENERATE AI ANSWER
# ============================================================

def generate_answer(
    question: str,
    context: str
) -> str:

    system_prompt = """
You are the official AI assistant for FLUX,
which stands for Future Leaders of Unbound Xperiments.

FLUX is a college organization at MMMUT.

Your task is to answer users' questions about FLUX
using the information retrieved from the official
FLUX website.

IMPORTANT RULES:

1. The website context provided to you is your
   primary source of truth.

2. Do not invent facts about FLUX.

3. Do not make up names, dates, events, positions,
   contact information, eligibility requirements,
   locations, or other details.

4. If the answer is present in the website context,
   answer it directly.

5. If the answer is not available in the provided
   website context, say that the information is not
   available in the FLUX website information you
   currently have.

6. Understand natural conversational language,
   informal language, abbreviations, and minor
   spelling mistakes.

7. If the user asks a simple question, give a
   concise answer.

8. If the user asks for an explanation, provide
   enough detail to make it understandable.

9. If the user asks multiple questions, answer
   all of them.

10. Never mention internal technologies such as
    ChromaDB, embeddings, RAG, retrieval, or
    vector databases unless the user specifically
    asks how the chatbot works.

11. Do not pretend that information is official
    unless it is supported by the provided website
    context.

12. If the user asks something unrelated to FLUX,
    you can answer naturally when appropriate,
    but make it clear that you are the FLUX assistant.

13. Do not unnecessarily repeat the source URL.

14. Answer naturally like a helpful human assistant.
"""

    user_prompt = f"""
Information retrieved from the official FLUX website:

================ WEBSITE INFORMATION ================

{context}

================ END WEBSITE INFORMATION ============


User's question:

{question}


Answer the user's question using the website
information above.
"""

    try:

        response = llm_client.chat.completions.create(

            model=HF_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            max_tokens=500,

            temperature=0.3
        )

        # Safely extract response
        if not response.choices:

            return (
                "I couldn't generate an answer "
                "right now."
            )

        answer = response.choices[0].message.content

        if not answer:

            return (
                "I couldn't generate an answer "
                "right now."
            )

        return answer.strip()

    except Exception as e:

        print()
        print("=" * 60)
        print("HUGGING FACE ERROR:")
        print(str(e))
        print("=" * 60)
        print()

        return (
            "I'm having trouble connecting to the AI "
            "service right now. Please try again in "
            "a moment."
        )


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post(
    "/chat",
    response_model=ChatResponse
)
async def chat(request: ChatRequest):

    question = request.message.strip()

    # Validate question
    if not question:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    print()
    print("=" * 60)

    print("USER QUESTION:")
    print(question)

    # --------------------------------------------------------
    # STEP 1: SEARCH WEBSITE
    # --------------------------------------------------------

    try:

        results = search_website(
            question,
            n_results=4
        )

    except Exception as e:

        print()
        print("CHROMADB ERROR:")
        print(str(e))

        raise HTTPException(
            status_code=500,
            detail="Knowledge base search failed."
        )


    # --------------------------------------------------------
    # STEP 2: DISPLAY SOURCES
    # --------------------------------------------------------

    print()
    print("RELEVANT SOURCES:")

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    if metadatas:

        for metadata in metadatas:

            title = metadata.get(
                "title",
                "Unknown"
            )

            url = metadata.get(
                "url",
                ""
            )

            print(
                f"- {title} | {url}"
            )

    else:

        print("- No sources found")


    # --------------------------------------------------------
    # STEP 3: BUILD CONTEXT
    # --------------------------------------------------------

    context = build_context(
        results
    )


    # --------------------------------------------------------
    # STEP 4: GENERATE AI ANSWER
    # --------------------------------------------------------

    answer = generate_answer(
        question,
        context
    )


    # --------------------------------------------------------
    # LOG ANSWER
    # --------------------------------------------------------

    print()
    print("AI ANSWER:")
    print(answer)

    print("=" * 60)


    # --------------------------------------------------------
    # RETURN RESPONSE
    # --------------------------------------------------------

    return ChatResponse(
        reply=answer
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "service": "FLUX AI Chatbot",
        "documents": collection.count(),
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": HF_MODEL,
        "huggingface_configured": bool(HF_TOKEN)
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "message": "FLUX AI Chatbot API",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )