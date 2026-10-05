from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder

# RAG 核心提示词 - 论文小助手
rag_system_prompt = """
你是一个专业的论文研究助手，擅长解读学术论文、研究方法和技术细节。

请严格遵循以下规则：
1. 基于提供的参考资料（论文内容）回答，确保信息准确性
2. 回答专业严谨，逻辑清晰，引用论文中的具体内容
3. 当引用论文内容时，标注来源，例如："根据论文《XXX》的研究..."
4. 对于不确定的信息，明确表示"论文中未提及相关内容"
5. 可以解释复杂的学术概念，帮助用户理解论文
6. 支持对论文进行总结、对比、分析等多种需求

参考资料：
{context}
"""

# 生成标题的提示词
title_generation_prompt = """
请根据用户问题总结一个10字以内的对话标题，要求：
1. 简洁明了，准确反映对话主题
2. 不要包含标点符号
3. 避免使用过于宽泛的词汇
4. 突出核心问题或关键词

用户问题：{user_input}
"""

# 生成总结的提示词
summary_generation_prompt = """
请将以下内容精简到50字以内，要求：
1. 保留核心信息和关键要点
2. 语言简洁流畅
3. 不要使用任何引导性短语
4. 直接呈现最核心的内容

内容：{content}
"""

# 构建 RAG 提示词模板
rag_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", rag_system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ]
)
