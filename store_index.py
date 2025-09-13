from dotenv import load_dotenv
import os
from pinecone import pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY= os.gotenv("PINECONE_API_KEY")
OPENAI_API_KEY= os.gotenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"]= PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]= OPENAI_API_KEY

extracted_data = load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunk =text_split(filter_data)

pinecone_api_key= PINECONE_API_KEY

pc=pinecone(api_key=pinecone_api_key)


index_name= "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name =index_name,
        dimension=384,    #index of the embedding
        metric= "cosine",    # cosine similarity
        spec= ServerlessSpec(cloud="aws",region="us-east-1")
    )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,
    index_name=index_name
)



