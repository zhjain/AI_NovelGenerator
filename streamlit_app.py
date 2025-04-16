import streamlit as st
import os
from novel_generator import (
    Novel_architecture_generate,
    Chapter_blueprint_generate,
    generate_chapter_draft,
    finalize_chapter,
    import_knowledge_file,
    clear_vector_store
)
from config_manager import load_config, save_config
import json

def init_session_state():
    """初始化session state变量"""
    if 'config' not in st.session_state:
        default_config = load_config('config.json') or {}
        st.session_state.config = default_config
    
    if 'current_chapter' not in st.session_state:
        st.session_state.current_chapter = 1

def main():
    st.set_page_config(page_title="AI Novel Generator", layout="wide")
    init_session_state()
    
    st.title("📚 AI小说生成器")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("配置设置")
        
        # LLM设置
        st.subheader("LLM模型设置")
        interface_format = st.selectbox(
            "接口类型",
            ["OpenAI", "Azure OpenAI", "Ollama", "ML Studio", "Gemini", "SiliconFlow"],
            key="interface_format"
        )
        
        api_key = st.text_input("API Key", type="password", key="api_key")
        base_url = st.text_input("Base URL", key="base_url")
        model_name = st.text_input("模型名称", key="model_name")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, key="temperature")
        
        # Embedding设置
        st.subheader("Embedding设置")
        embedding_format = st.selectbox(
            "Embedding接口",
            ["OpenAI", "Azure OpenAI", "Ollama", "SiliconFlow"],
            key="embedding_format"
        )
        
        # 保存配置按钮
        if st.button("保存配置"):
            config = {
                "interface_format": st.session_state.interface_format,
                "api_key": st.session_state.api_key,
                "base_url": st.session_state.base_url,
                "model_name": st.session_state.model_name,
                "temperature": st.session_state.temperature,
                # ... 其他配置项
            }
            save_config("config.json", config)
            st.success("配置已保存!")

    # 主要内容区域
    tab1, tab2, tab3, tab4 = st.tabs(["小说设定", "章节生成", "角色库", "知识库"])
    
    with tab1:
        st.header("小说基础设定")
        topic = st.text_area("小说主题")
        genre = st.text_input("类型")
        num_chapters = st.number_input("章节数", min_value=1, value=10)
        word_number = st.number_input("每章字数", min_value=1000, value=3000)
        
        if st.button("生成小说架构"):
            with st.spinner("正在生成小说架构..."):
                try:
                    Novel_architecture_generate(
                        interface_format=st.session_state.interface_format,
                        api_key=st.session_state.api_key,
                        base_url=st.session_state.base_url,
                        llm_model=st.session_state.model_name,
                        topic=topic,
                        genre=genre,
                        number_of_chapters=num_chapters,
                        word_number=word_number,
                        filepath="./output",
                        temperature=st.session_state.temperature
                    )
                    st.success("小说架构生成完成!")
                except Exception as e:
                    st.error(f"生成失败: {str(e)}")
    
    with tab2:
        st.header("章节生成")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            chapter_content = st.text_area(
                "章节内容",
                height=400,
                key="chapter_content"
            )
            
        with col2:
            st.subheader("操作")
            if st.button("生成大纲"):
                with st.spinner("正在生成章节大纲..."):
                    try:
                        Chapter_blueprint_generate(
                            interface_format=st.session_state.interface_format,
                            api_key=st.session_state.api_key,
                            base_url=st.session_state.base_url,
                            llm_model=st.session_state.model_name,
                            filepath="./output",
                            number_of_chapters=num_chapters
                        )
                        st.success("章节大纲生成完成!")
                    except Exception as e:
                        st.error(f"生成失败: {str(e)}")
            
            if st.button("生成草稿"):
                # 实现章节草稿生成逻辑
                pass
                
            if st.button("完善章节"):
                # 实现章节完善逻辑
                pass

    with tab3:
        st.header("角色库")
        # 实现角色库功能
        
    with tab4:
        st.header("知识库")
        uploaded_file = st.file_uploader("上传知识库文件", type=['txt', 'md', 'pdf'])
        if uploaded_file is not None:
            # 处理上传的知识库文件
            pass

if __name__ == "__main__":
    main()