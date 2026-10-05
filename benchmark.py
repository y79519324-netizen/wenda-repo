"""
RAG 检索量化评估脚本
====================
评估三种检索策略的 Recall@1/3/5、Precision@3、MRR 指标：
  1. Dense-only（单向量检索）
  2. Sparse-only（单BM25关键词检索）
  3. Hybrid RRF（混合检索，当前项目默认）

用法：
  # 1. 先准备你的知识库（上传一些文档到 Milvus/BM25）
  # 2. 运行评估：
  python benchmark.py

原理（离线检索评估，不需要 LLM 判断）：
  对每个 query 手动标注 "正确文档里应该包含哪些关键词"（golden_keywords），
  如果检索返回的文档包含了所有 golden_keywords，就认为该文档是"相关文档"。
  （这是一种简化的"弱标注"，不需要人工标注完整的 golden_documents）
"""

import os
import sys
from typing import List, Dict, Any

# ---------- 1. 加载项目模块 ----------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.vector_stores import VectorStoreService
from app.core import config_data as config


# ---------- 2. 测试用例（请根据你实际的知识库内容修改） ----------
# 说明：
#   question: 用户的问题
#   golden_keywords: 正确答案文档中"必须全部出现"的关键词
#                    只要检索到的某文档包含了所有这些关键词 → 认为该文档相关
#
# ⚠️ 请根据你实际上传到知识库的内容修改！
#    如果知识库还没有内容，先通过 API 上传 3-5 篇 PDF/DOCX 再测
TEST_CASES = [
    {
        "question": "什么是 RRF 倒数秩融合算法？",
        "golden_keywords": ["RRF", "倒数秩融合", "rank"],
    },
    {
        "question": "Milvus Lite 和 Milvus Server 的区别是什么？",
        "golden_keywords": ["Milvus", "Lite", "嵌入式", "Server"],
    },
    {
        "question": "SemanticChunker 语义分割和固定长度切分有什么区别？",
        "golden_keywords": ["SemanticChunker", "语义", "切分"],
    },
    {
        "question": "BM25 算法的核心思想是什么？",
        "golden_keywords": ["BM25", "TF", "IDF", "词频"],
    },
    {
        "question": "LangChain LCEL 是什么？",
        "golden_keywords": ["LCEL", "LangChain", "表达式", "chain"],
    },
    {
        "question": "FastAPI 流式响应 SSE 和 WebSocket 的区别？",
        "golden_keywords": ["流式", "SSE", "WebSocket", "HTTP"],
    },
    {
        "question": "向量检索中余弦距离和欧氏距离的区别？",
        "golden_keywords": ["余弦", "欧氏", "距离", "相似度"],
    },
    {
        "question": "MD5 去重的原理是什么？",
        "golden_keywords": ["MD5", "哈希", "去重", "指纹"],
    },
    {
        "question": "MySQL 中 InnoDB 和 MyISAM 的区别？",
        "golden_keywords": ["InnoDB", "MyISAM", "事务", "索引"],
    },
    {
        "question": "RAG 系统如何缓解大模型幻觉？",
        "golden_keywords": ["RAG", "检索", "幻觉", "参考资料"],
    },
]
# 注意：如果你的知识库内容和上面这些问题不匹配，
#       请把 TEST_CASES 改成你知识库中实际存在的内容相关问题。
#       否则 Recall 会是 0，但这是测试用例的问题，不是检索系统的问题。


# ---------- 3. 三种检索器 ----------
def build_dense_retriever(service: VectorStoreService):
    """单路 Dense 检索（Milvus 向量）"""
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from typing import List

    class DenseRetriever(BaseRetriever):
        k: int = 3

        def _get_relevant_documents(self, query: str) -> List[Document]:
            return service.search_milvus(query, k=self.k)

    return DenseRetriever(k=5)


def build_sparse_retriever(service: VectorStoreService):
    """单路 Sparse 检索（BM25）"""
    r = service._get_bm25_retriever()
    if r:
        r.k = 5
    return r


def build_hybrid_retriever(service: VectorStoreService):
    """混合检索 RRF（项目当前默认）"""
    return service.get_retriever()


# ---------- 4. 判定"文档是否相关" ----------
def is_relevant(doc_content: str, golden_keywords: List[str]) -> bool:
    """
    简化判定：文档内容是否包含了所有 golden_keywords（忽略大小写）
    更严谨的做法是用 LLM-as-Judge，但那样评估慢且要花钱。
    """
    content = doc_content.lower()
    return all(kw.lower() in content for kw in golden_keywords)


# ---------- 5. 指标计算 ----------
def calc_metrics(retrieved_docs, golden_keywords, k_list=[1, 3, 5]):
    """
    计算一组检索结果的指标：
    - Recall@k：前 k 个结果中，"相关文档数 / 实际相关文档总数"
                （简化版：如果前k个里至少有1个相关，就认为命中）
    - Precision@k：前 k 个结果中，相关文档的占比
    - MRR：第一个相关文档的排名倒数（Mean Reciprocal Rank）
    """
    # 判定每个文档是否相关
    relevance_flags = []
    for doc in retrieved_docs:
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        relevance_flags.append(is_relevant(content, golden_keywords))

    metrics = {}

    # Recall@k（Hit Rate 简化版：前k个里至少有1个相关就算命中）
    for k in k_list:
        top_k = relevance_flags[:k]
        metrics[f"Recall@{k}"] = 1.0 if any(top_k) else 0.0

    # Precision@k（前k个里相关文档的比例）
    # 注意：Precision 的分母是 k，所以如果检索返回不满 k 条，也要按 k 算
    for k in k_list:
        top_k = relevance_flags[:k]
        relevant_in_top = sum(top_k)
        metrics[f"Precision@{k}"] = relevant_in_top / k if k > 0 else 0.0

    # MRR（第一个相关文档出现位置的倒数）
    mrr = 0.0
    for idx, is_rel in enumerate(relevance_flags, 1):
        if is_rel:
            mrr = 1.0 / idx
            break
    metrics["MRR"] = mrr

    # 实际相关文档总数（用于 Debug）
    metrics["_relevant_count"] = sum(relevance_flags)
    metrics["_retrieved_count"] = len(retrieved_docs)

    return metrics


# ---------- 6. 主评估流程 ----------
def run_benchmark():
    print("=" * 70)
    print("  RAG 检索量化评估")
    print("=" * 70)

    # 1. 初始化检索服务
    print("\n[1/4] 初始化 VectorStoreService...")
    try:
        service = VectorStoreService()
    except Exception as e:
        print(f"  ❌ 初始化失败: {e}")
        print("     请确保 Milvus 数据库文件存在，且没有其他进程占用")
        return

    # 2. 构建三种检索器
    print("\n[2/4] 构建三种检索器...")
    dense_ret = build_dense_retriever(service)
    sparse_ret = build_sparse_retriever(service)
    hybrid_ret = build_hybrid_retriever(service)

    retrievers = {
        "Dense (向量)": dense_ret,
    }
    if sparse_ret:
        retrievers["Sparse (BM25)"] = sparse_ret
    if hybrid_ret:
        retrievers["Hybrid (RRF)"] = hybrid_ret

    print(f"  可用检索器: {list(retrievers.keys())}")

    # 3. 运行所有测试用例
    print(f"\n[3/4] 运行 {len(TEST_CASES)} 条测试用例...")
    print("-" * 70)

    # 存储每条结果
    all_results = {name: [] for name in retrievers.keys()}

    for i, case in enumerate(TEST_CASES):
        question = case["question"]
        keywords = case["golden_keywords"]
        print(f"\n  [{i+1}/{len(TEST_CASES)}] {question}")
        print(f"         关键词: {keywords}")

        for name, ret in retrievers.items():
            try:
                docs = ret.invoke(question)
                metrics = calc_metrics(docs, keywords)
                all_results[name].append(metrics)

                # 打印简表
                r1 = f"R@1={metrics['Recall@1']:>4.0%}"
                r3 = f"R@3={metrics['Recall@3']:>4.0%}"
                mrr = f"MRR={metrics['MRR']:.2f}"
                rel = f"相关={metrics['_relevant_count']}/{metrics['_retrieved_count']}"
                print(f"    {name:<16}: {r1}  {r3}  {mrr}  ({rel})")
            except Exception as e:
                print(f"    {name:<16}: 错误 {e}")

    # 4. 汇总统计
    print("\n" + "=" * 70)
    print("[4/4] 汇总结果（全部测试用例的平均值）")
    print("=" * 70)

    header = f"{'检索策略':<16} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'P@3':>6} {'P@5':>6} {'MRR':>6}"
    print("\n" + header)
    print("-" * len(header))

    summary = {}
    for name, results in retrievers.items():
        if not all_results[name]:
            continue
        # 取平均
        n = len(all_results[name])
        avg = {}
        for key in ["Recall@1", "Recall@3", "Recall@5",
                    "Precision@3", "Precision@5", "MRR"]:
            avg[key] = sum(r[key] for r in all_results[name]) / n

        summary[name] = avg
        line = (f"{name:<16} "
                f"{avg['Recall@1']:>6.1%} "
                f"{avg['Recall@3']:>6.1%} "
                f"{avg['Recall@5']:>6.1%} "
                f"{avg['Precision@3']:>6.1%} "
                f"{avg['Precision@5']:>6.1%} "
                f"{avg['MRR']:>6.2f}")
        print(line)

    # 5. 混合检索对比单路的提升
    print("\n" + "-" * 70)
    print("  Hybrid 对比单路的提升：")
    if "Hybrid (RRF)" in summary and "Dense (向量)" in summary:
        for metric in ["Recall@3", "MRR"]:
            delta = summary["Hybrid (RRF)"][metric] - summary["Dense (向量)"][metric]
            pct = (delta / summary["Dense (向量)"][metric] * 100
                   if summary["Dense (向量)"][metric] > 0 else 0)
            print(f"    {metric:<12}: {summary['Dense (向量)'][metric]:.1%} → "
                  f"{summary['Hybrid (RRF)'][metric]:.1%}  "
                  f"(+{pct:.1f}%)")

    # 6. 结论建议
    print("\n" + "=" * 70)
    if "Hybrid (RRF)" in summary:
        r3_hybrid = summary["Hybrid (RRF)"]["Recall@3"]
        if r3_hybrid >= 0.7:
            print(f"  ✅ 混合检索 Recall@3 = {r3_hybrid:.1%}，效果良好")
        elif r3_hybrid >= 0.5:
            print(f"  ⚠️  混合检索 Recall@3 = {r3_hybrid:.1%}，有提升空间，建议：")
            print("      - 增加 BM25 k 值 (当前 3，可调到 5)")
            print("      - 调优 RRF 的 k 参数 (当前 60)")
            print("      - 增加更多优质文档到知识库")
        else:
            print(f"  ❌ 混合检索 Recall@3 = {r3_hybrid:.1%}，偏低，建议：")
            print("      - 检查 TEST_CASES 是否和知识库内容匹配")
            print("      - 确认 golden_keywords 标注是否准确")
            print("      - 评估语义分割 chunk_size 是否过大/过小")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
