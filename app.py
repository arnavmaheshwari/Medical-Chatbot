import os
from dotenv import load_dotenv
from flask import Flask, render_template, request

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import FastEmbedEmbeddings

from src.prompt import MEDICAL_QA_PROMPT


# -------------------- App Setup --------------------

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# -------------------- Pinecone Setup --------------------

pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "medical-chatbot"

embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
    pool_threads=1
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# -------------------- LLM Setup --------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-robotics-er-1.5-preview",
    temperature=0.5,
    api_key=GEMINI_API_KEY
)


# -------------------- RAG Logic --------------------

def retrieval_qa_with_sources(llm, retriever, question):
    docs = retriever.invoke(question)

    if not docs:
        return {
            "answer": "I don't know. No relevant medical information was found."
        }

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = MEDICAL_QA_PROMPT.format(
        context=context,
        question=question
    )

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# -------------------- Routes --------------------

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_message = request.form["msg"]

    result = retrieval_qa_with_sources(
        llm=llm,
        retriever=retriever,
        question=user_message
    )

    return result["answer"]


# -------------------- Main --------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)