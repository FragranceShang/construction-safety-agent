from graph.graph import build_regulation_graph

def main():
    regulation_graph = build_regulation_graph()
    result = regulation_graph.invoke({
        "question": "雨季施工需要注意哪些用电安全事项？",
        # "retriever": retriever,
    })

    print(result["answer"])

if __name__ == "__main__":
    main()
