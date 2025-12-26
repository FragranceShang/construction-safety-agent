import os
from mem0 import Memory
from mem0.configs.base import MemoryConfig

def init_mem0():
    os.environ["DEEPSEEK_API_KEY"] = "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"
    os.environ["OPENAI_API_KEY"] = "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"

    return Memory(
        MemoryConfig(
            embedder={
                "provider": "huggingface",
                "config": {
                    "model": "BAAI/bge-base-zh"
                }
            },
            vector_store={
                "provider": "faiss",
                "config": {
                    "path": "db/mem0_data/faiss",
                    "collection_name": "regulation_memory"
                }
            },
            llm={
                "provider": "deepseek",
                "config": {
                    "model": "nex-agi/deepseek-v3.1-nex-n1:free",
                    "deepseek_base_url": "https://openrouter.ai/api/v1"
                }
            },
            history_db_path="db/mem0_data/history.db"
        )
    )
