import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from src.helper import download_hugging_face_embeddings
from src.prompt import MEDICAL_QA_PROMPT



# -------------------- App Setup --------------------

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# -------------------- Pinecone Setup --------------------

pc = Pinecone(api_key=PINECONE_API_KEY)

embeddings = download_hugging_face_embeddings()

INDEX_NAME = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# -------------------- LLM Setup --------------------

llm = ChatGoogleGenerativeAI(
            model="gemini-robotics-er-1.5-preview",
            temperature=0.5,
            api_key=os.getenv("GEMINI_API_KEY")
        )


# -------------------- RAG Logic --------------------

def retrieval_qa_with_sources(llm, retriever, question):
    docs = retriever.invoke(question)

    if not docs:
        return {
            "answer": "I don't know. No relevant medical information was found.",
        }

    context_blocks = []

    for doc in docs:
        context_blocks.append(doc.page_content)

    context = "\n\n".join(context_blocks)

    prompt = MEDICAL_QA_PROMPT.format(
        context=context,
        question=question
    )

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
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
    app.run(host="0.0.0.0", port=8080, debug=True)