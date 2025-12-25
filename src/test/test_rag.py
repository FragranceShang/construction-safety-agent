from ..rag import answer_regulation


if __name__ == "__main__":
    from ..graph.state.retriever import load_vectorstore, build_retriever
    vectorstore = load_vectorstore("faiss_index")
    retriever = build_retriever(vectorstore, top_k=9)
    with open("test_results.txt", "w", encoding="utf-8") as f:

        f.write("\n===测试基本检索能力===\n")
        for question in [
            "配电箱的安装高度有什么要求？",
            "架空线路跨越道路时的最小垂直距离是多少？",
        ]:
            result = answer_regulation(
                question=question,
                retriever=retriever,
            )
            f.write("【问题】\n")
            f.write(question)
            f.write("\n【回答】\n")
            f.write(result)
            f.write("\n")

        f.write("\n===测试理解和解释能力===\n")
        for question in [
            "为什么行灯变压器不能带入金属容器内使用？",
            "多级剩余电流保护器如何设置动作电流和时间？",
        ]:
            result = answer_regulation(
                question=question,
                retriever=retriever,
            )
            f.write("【问题】\n")
            f.write(question)
            f.write("\n【回答】\n")
            f.write(result)
            f.write("\n")

        f.write("\n===测试实际应用能力===\n")
        for question in [
            "在脚手架上作业时，对外电线路的安全距离有什么要求？",
            "雨季施工需要注意哪些用电安全事项？",
            "塔式起重机接地应该如何设置？",
        ]:
            result = answer_regulation(
                question=question,
                retriever=retriever,
            )
            f.write("【问题】\n")
            f.write(question)
            f.write("\n【回答】\n")
            f.write(result)
            f.write("\n")

        f.write("\n===测试深度分析能力===\n")
        for question in [
            "手持式电动工具在一般场所和潮湿场所的要求有什么不同？",
            "临时照明和行灯在电压要求上有何区别？为什么？",
            "易燃易爆环境和腐蚀环境对电缆敷设要求有何异同？",
        ]:
            result = answer_regulation(
                question=question,
                retriever=retriever,
            )
            f.write("【问题】\n")
            f.write(question)
            f.write("\n【回答】\n")
            f.write(result)
            f.write("\n")

        f.write("\n===测试实际工作应用===\n")
        for question in [
            "检查配电箱时需要关注哪些关键点？",
            "直埋电缆施工需要满足哪些技术要求？",
            "外电线路附近施工应采取哪些防护措施？",
        ]:
            result = answer_regulation(
                question=question,
                retriever=retriever,
            )
            f.write("【问题】\n")
            f.write(question)
            f.write("\n【回答】")
            f.write(result)
            f.write("\n")

        f.write("\n===测试系统健壮性===\n")
        for question in [
            "本规范适用于井下工程吗",
            "规范中对高原环境的特殊要求是什么？",
            "焊接机械的二次线为什么不能用钢筋代替？",
        ]:
            result = answer_regulation(
                question=question,
                retriever=retriever,
            )
            f.write("【问题】\n")
            f.write(question)
            f.write("\n【回答】\n")
            f.write(result)
            f.write("\n")