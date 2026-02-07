import os
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings, load_pdf_files, text_split, filter_to_minimal_docs


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)


extracted_data = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extracted_data)
chunks = text_split(minimal_docs)
embeddings = download_hugging_face_embeddings()


#Creating index in pinecone
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension=384,
        metric= "cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

#Storing vector into pinecone vector db
docsearch = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name
)