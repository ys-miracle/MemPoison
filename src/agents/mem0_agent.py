from mem0 import Memory

class Mem0Agent:
    def __init__(self, db_path="./mem0_db"):
        config = {
            "vector_store": {"provider": "chroma", "config": {"collection_name": "attack_test", "path": db_path}},
            "embedder": {"provider": "huggingface", "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"}},
            "llm": {"provider": "openai", "config": {"model": "gpt-4o", "temperature": 0.4}}
        }
        self.memory = Memory.from_config(config)

    def inject(self, text, user_id):
        self.memory.add(text, user_id=user_id)
        all_mems = self.memory.get_all(user_id=user_id, limit=1)
        return all_mems[0]['id'], all_mems[0]['memory'], []

    def search(self, query, user_id, k=1):
        results = self.memory.search(query, user_id=user_id, limit=k)
        return [m['memory'] for m in results if 'memory' in m]

    def delete_all(self, user_id):
        self.memory.delete_all(user_id=user_id)