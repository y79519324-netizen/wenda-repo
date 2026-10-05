"""
基于Streamlit完成WEB网页上传服务

支持的文件格式: .txt, .docx, .pdf, .md, .csv, .json

当WEB页面发生变化（刷新 上传）则代码重新执行一遍，因此要用seesion_state
"""
import time
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from app.core.knowledge_base import KnowledgeBaseService
from app.core.file_parser import FileParser

# 添加网页标题
st.title("知识库更新服务")

# 初始化会话状态
if 'service' not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

# 支持的文件类型
supported_types = FileParser.get_supported_extensions()
type_hint = FileParser.get_supported_type_hint()

# 添加文件上传服务
upload_file = st.file_uploader(
    f"请上传文件（支持 {type_hint}）",
    type=supported_types,
    accept_multiple_files=False  # 仅接受一个文件的上传
)

if upload_file:
    file_name = upload_file.name
    file_size = upload_file.size / 1024
    file_type = upload_file.type or os.path.splitext(file_name)[1]
    
    st.subheader(f"📄 文件名: {file_name}")
    st.write(f"📎 格式: {file_type} | 📏 大小: {file_size:.2f} KB")
    
    # 获取文件的二进制内容
    file_content = upload_file.getvalue()
    
    try:
        # 根据文件类型解析内容
        with st.spinner("📖 正在解析文件内容..."):
            text = FileParser.parse_file(file_content, file_name)
        
        # 显示解析结果预览
        with st.expander("📋 预览解析后的内容（前 500 字）"):
            preview = text[:500] if len(text) > 500 else text
            st.text(preview + ("..." if len(text) > 500 else ""))
            st.info(f"📊 文档总长度: {len(text)} 字符")
        
        # 确认上传按钮
        if st.button("✅ 确认上传到知识库", type="primary"):
            with st.spinner("🔄 正在载入知识库中..."):
                time.sleep(0.5)
                result = st.session_state["service"].upload_by_str(text, file_name)
            
            # 显示结果
            if "成功" in result:
                st.success(result)
            elif "跳过" in result:
                st.warning(result)
            else:
                st.error(result)
                
    except ValueError as e:
        st.error(f"❌ {str(e)}")
    except Exception as e:
        st.error(f"❌ 上传失败: {str(e)}")

# 添加使用说明
with st.expander("📖 使用说明"):
    st.markdown("""
    ### 支持的文件格式
    
    | 格式 | 扩展名 | 说明 |
    |------|--------|------|
    | 纯文本 | .txt | 普通文本文件 |
    | Word 文档 | .docx | Microsoft Word 文档（支持段落和表格） |
    | PDF 文档 | .pdf | Adobe PDF 文档 |
    | Markdown | .md | Markdown 格式文件 |
    | CSV 表格 | .csv | 逗号分隔值文件 |
    | JSON | .json | JavaScript 对象表示法 |
    
    ### 使用步骤
    
    1. 点击上方的文件上传按钮
    2. 选择一个支持的文件
    3. 预览解析后的内容
    4. 点击"确认上传到知识库"按钮
    
    ### 注意事项
    
    - 首次上传会自动创建知识库
    - 重复内容会被自动跳过（基于 MD5 去重）
    - 文件内容会被语义分割后存入向量数据库
    """)
