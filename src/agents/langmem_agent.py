from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langmem import create_memory_manager

class UserFact(BaseModel):
    fact: str = Field(description="A single extracted valuable fact.")
class FactList(BaseModel):
    facts: list[UserFact] = Field(description="A list of facts.")

class LangMemAgent:
    def __init__(self, db_dir):
        llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = Chroma(collection_name="eval_memory", embedding_function=self.embeddings, persist_directory=db_dir)
        self.extractor = create_memory_manager(llm, schemas=[FactList], instructions="Extract personal facts.")

    def search(self, query, k=1):
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def delete(self, ids):
        self.vector_store.delete(ids=ids)