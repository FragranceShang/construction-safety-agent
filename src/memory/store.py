from langgraph.store.memory import InMemoryStore
from langmem import create_memory_searcher

from src.graph.state.answer import get_llm

memory_store = InMemoryStore()
memory_store.put(
    namespace=("regulation", "GB50016"),
    key="3.1.2",
    value={
        "text": "建筑物的耐火等级不应低于二级",
        "article": "3.1.2"
    }
)

# 3. Searcher（LangMem）
llm = get_llm()

searcher = create_memory_searcher(
    llm,
    namespace=("regulation", "GB50016")
)

# 4. 搜索
results = searcher.invoke({
    "messages": [
        {"role": "user", "content": "建筑耐火等级有什么要求？"}
    ]
})

for r in results:
    print(r.value, r.score)