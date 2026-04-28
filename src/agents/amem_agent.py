import os
import pickle
from agentic_memory.memory_system import AgenticMemorySystem
from agentic_memory.retrievers import PersistentChromaRetriever

class AMemAgent:
    def __init__(self, base_db_dir, llm_model="gpt-4o"):
        self.memory_sys = AgenticMemorySystem(
            model_name='all-MiniLM-L6-v2',  
            llm_backend="openai",           
            llm_model=llm_model        
        )
        os.makedirs(base_db_dir, exist_ok=True)
        self.memory_sys.retriever = PersistentChromaRetriever(
            directory=base_db_dir, collection_name="memories", extend=True
        )

    def inject(self, text):
        extracted = self.memory_sys.analyze_content(text)
        content = extracted.get("context", text)
        keywords = extracted.get("keywords", [])
        tags = extracted.get("tags", [])
        m_id = self.memory_sys.add_note(content=content, keywords=keywords, context=content, tags=tags)
        return m_id, content, keywords

    def search(self, query, k=1):
        return self.memory_sys.search_agentic(query, k=k)

    def delete(self, m_id):
        self.memory_sys.delete(m_id)