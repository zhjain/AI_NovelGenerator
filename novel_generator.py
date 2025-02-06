# novel_generator.py
# -*- coding: utf-8 -*-
import os
import logging
import re
import time
import traceback
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain.docstore.document import Document

# nltk、sentence_transformers 及文本处理相关
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 工具函数
from utils import (
    read_file, append_text_to_file, clear_file_content,
    save_string_to_txt
)

# prompt模板
from prompt_definitions import (
    core_seed_prompt,
    character_dynamics_prompt,
    world_building_prompt,
    plot_architecture_prompt,
    chapter_blueprint_prompt,
    summary_prompt,
    update_character_state_prompt,
    chapter_draft_prompt,
    summarize_recent_chapters_prompt
)

# 章节目录解析
from chapter_directory_parser import get_chapter_info_from_blueprint

from llm_adapters import create_llm_adapter
from embedding_adapters import create_embedding_adapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ============ 工具函数 ============

def remove_think_tags(text: str) -> str:
    """移除 <think>...</think> 包裹的内容"""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def debug_log(prompt: str, response_content: str):
    logging.info(
        f"\n[#########################################  Prompt  #########################################]\n{prompt}\n"
    )
    logging.info(
        f"\n[######################################### Response #########################################]\n{response_content}\n"
    )

def invoke_with_cleaning(llm_adapter, prompt: str) -> str:
    """通用封装：调用 LLM，并移除 <think>...</think> 文本，记录日志后返回"""
    response = llm_adapter.invoke(prompt)
    if not response:
        logging.warning("No response from model.")
        return ""
    cleaned_text = remove_think_tags(response)
    debug_log(prompt, cleaned_text)
    return cleaned_text.strip()

# ============ 获取 vectorstore 路径 ============

def get_vectorstore_dir(filepath: str) -> str:
    return os.path.join(filepath, "vectorstore")

# ============ 清空向量库 ============

def clear_vector_store(filepath: str) -> bool:
    import shutil
    store_dir = get_vectorstore_dir(filepath)
    if not os.path.exists(store_dir):
        logging.info("No vector store found to clear.")
        return False
    try:
        shutil.rmtree(store_dir)
        logging.info(f"Vector store directory '{store_dir}' removed.")
        return True
    except Exception as e:
        logging.error(f"无法删除向量库文件夹，请关闭程序后手动删除 {store_dir}。\n {str(e)}")
        traceback.print_exc()
        return False

# ============ 根据 embedding 接口创建/加载 Chroma ============

def init_vector_store(
    embedding_adapter,
    texts: List[str],
    filepath: str
) -> Chroma:
    """
    在 filepath 下创建/加载一个 Chroma 向量库并插入 texts。
    这里 embedding_adapter 是一个实现了 embed_documents(texts) 的对象
    """
    store_dir = get_vectorstore_dir(filepath)
    os.makedirs(store_dir, exist_ok=True)

    # 将文本封装为 Document
    documents = [Document(page_content=str(t)) for t in texts]

    # 因为我们是自定义的 embeddings，对接Chroma时需包装一个“langchain兼容对象”
    # 这里示例：写一个包装函数
    from langchain.embeddings.base import Embeddings as LCEmbeddings

    class LCEmbeddingWrapper(LCEmbeddings):
        def embed_documents(self, doc_texts: List[str]) -> List[List[float]]:
            return embedding_adapter.embed_documents(doc_texts)

        def embed_query(self, query_text: str) -> List[float]:
            return embedding_adapter.embed_query(query_text)

    chroma_embedding = LCEmbeddingWrapper()

    vectorstore = Chroma.from_documents(
        documents,
        embedding=chroma_embedding,
        persist_directory=store_dir,
        client_settings=Settings(anonymized_telemetry=False),
        collection_name="novel_collection"
    )
    return vectorstore

def load_vector_store(
    embedding_adapter,
    filepath: str
) -> Optional[Chroma]:
    """
    读取已存在的 Chroma 向量库。若不存在则返回 None。
    """
    store_dir = get_vectorstore_dir(filepath)
    if not os.path.exists(store_dir):
        logging.info("Vector store not found. Will return None.")
        return None

    # 同样要包装embedding_adapter
    from langchain.embeddings.base import Embeddings as LCEmbeddings

    class LCEmbeddingWrapper(LCEmbeddings):
        def embed_documents(self, doc_texts: List[str]) -> List[List[float]]:
            return embedding_adapter.embed_documents(doc_texts)

        def embed_query(self, query_text: str) -> List[float]:
            return embedding_adapter.embed_query(query_text)

    chroma_embedding = LCEmbeddingWrapper()

    return Chroma(
        persist_directory=store_dir,
        embedding_function=chroma_embedding,
        client_settings=Settings(anonymized_telemetry=False),
        collection_name="novel_collection"
    )

# ============ 文本分段工具 ============

def split_by_length(text: str, max_length: int = 500) -> List[str]:
    segments = []
    start_idx = 0
    while start_idx < len(text):
        end_idx = min(start_idx + max_length, len(text))
        segment = text[start_idx:end_idx]
        segments.append(segment.strip())
        start_idx = end_idx
    return segments

def split_text_for_vectorstore(chapter_text: str,
                               max_length: int = 500,
                               similarity_threshold: float = 0.7) -> List[str]:
    """
    对新的章节文本进行分段后，再用于存入向量库。
    先句子切分 -> 语义相似度合并 -> 再按 max_length 切分。
    """
    if not chapter_text.strip():
        return []

    nltk.download('punkt', quiet=True)
    sentences = nltk.sent_tokenize(chapter_text)
    if not sentences:
        return []

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(sentences)

    merged_paragraphs = []
    current_sentences = [sentences[0]]
    current_embedding = embeddings[0]

    for i in range(1, len(sentences)):
        sim = cosine_similarity([current_embedding], [embeddings[i]])[0][0]
        if sim >= similarity_threshold:
            current_sentences.append(sentences[i])
            current_embedding = (current_embedding + embeddings[i]) / 2.0
        else:
            merged_paragraphs.append(" ".join(current_sentences))
            current_sentences = [sentences[i]]
            current_embedding = embeddings[i]

    if current_sentences:
        merged_paragraphs.append(" ".join(current_sentences))

    final_segments = []
    for para in merged_paragraphs:
        if len(para) > max_length:
            sub_segments = split_by_length(para, max_length=max_length)
            final_segments.extend(sub_segments)
        else:
            final_segments.append(para)

    return final_segments

# ============ 更新向量库 ============

def update_vector_store(
    embedding_adapter,
    new_chapter: str,
    filepath: str
):
    """
    将最新章节文本插入到向量库中。若库不存在则初始化。
    """
    splitted_texts = split_text_for_vectorstore(new_chapter)
    if not splitted_texts:
        logging.warning("No valid text to insert into vector store. Skipping.")
        return

    store = load_vector_store(embedding_adapter, filepath)
    if not store:
        logging.info("Vector store does not exist. Initializing a new one for new chapter...")
        init_vector_store(embedding_adapter, splitted_texts, filepath)
        return

    docs = [Document(page_content=str(t)) for t in splitted_texts]
    store.add_documents(docs)
    logging.info("Vector store updated with the new chapter splitted segments.")
    
# ============ 向量检索上下文 ============

def get_relevant_context_from_vector_store(
    embedding_adapter,
    query: str,
    filepath: str,
    k: int = 2
) -> str:
    """
    从向量库中检索与 query 最相关的 k 条文本，拼接后返回。
    """
    store = load_vector_store(embedding_adapter, filepath)
    if not store:
        logging.info("No vector store found. Returning empty context.")
        return ""

    docs = store.similarity_search(query, k=k)
    if not docs:
        logging.info(f"No relevant documents found for query '{query}'. Returning empty context.")
        return ""

    combined = "\n".join([d.page_content for d in docs])
    return combined

# ============ 从目录中获取最近 n 章文本 ============

def get_last_n_chapters_text(chapters_dir: str, current_chapter_num: int, n: int = 3) -> List[str]:
    texts = []
    start_chap = max(1, current_chapter_num - n)
    for c in range(start_chap, current_chapter_num):
        chap_file = os.path.join(chapters_dir, f"chapter_{c}.txt")
        if os.path.exists(chap_file):
            text = read_file(chap_file).strip()
            texts.append(text)
        else:
            texts.append("")
    return texts

# ============ 提炼(短期摘要, 下一章关键字) ============

def summarize_recent_chapters(
    interface_format: str,
    api_key: str,
    base_url: str,
    model_name: str,
    temperature: float,
    chapters_text_list: List[str]
) -> Tuple[str, str]:
    """
    生成 (short_summary, next_chapter_keywords)
    如果解析失败，则返回 (合并文本, "")
    """
    combined_text = "\n".join(chapters_text_list).strip()
    if not combined_text:
        return ("", "")

    # 1) 构造 llm_adapter
    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature
    )

    prompt = summarize_recent_chapters_prompt.format(combined_text=combined_text)
    response_text = invoke_with_cleaning(llm_adapter, prompt)

    short_summary = ""
    next_chapter_keywords = ""

    for line in response_text.splitlines():
        line = line.strip()
        if line.startswith("短期摘要:"):
            short_summary = line.replace("短期摘要:", "").strip()
        elif line.startswith("下一章关键字:"):
            next_chapter_keywords = line.replace("下一章关键字:", "").strip()

    if not short_summary and not next_chapter_keywords:
        short_summary = response_text

    return (short_summary, next_chapter_keywords)


# ============ 1) 生成总体架构 ============

def Novel_architecture_generate(
    api_key: str,
    base_url: str,
    llm_model: str,
    topic: str,
    genre: str,
    number_of_chapters: int,
    word_number: int,
    filepath: str,
    temperature: float = 0.7
) -> None:
    """
    依次调用:
      1. core_seed_prompt
      2. character_dynamics_prompt
      3. world_building_prompt
      4. plot_architecture_prompt
    最终输出 Novel_architecture.txt
    """
    os.makedirs(filepath, exist_ok=True)

    # 通过工厂函数创建 LLM 适配器
    llm_adapter = create_llm_adapter(
        interface_format="openai",  # 或根据你的实际：若你在UI中就是 "OpenAI" 就传递过来
        base_url=base_url,
        model_name=llm_model,
        api_key=api_key,
        temperature=temperature
    )

    # Step1: 核心种子
    prompt_core = core_seed_prompt.format(
        topic=topic,
        genre=genre,
        number_of_chapters=number_of_chapters,
        word_number=word_number
    )
    core_seed_result = invoke_with_cleaning(llm_adapter, prompt_core)

    # Step2: 角色动力学
    prompt_character = character_dynamics_prompt.format(core_seed=core_seed_result.strip())
    character_dynamics_result = invoke_with_cleaning(llm_adapter, prompt_character)

    # Step3: 世界观
    prompt_world = world_building_prompt.format(core_seed=core_seed_result.strip())
    world_building_result = invoke_with_cleaning(llm_adapter, prompt_world)

    # Step4: 三幕式情节
    prompt_plot = plot_architecture_prompt.format(
        core_seed=core_seed_result.strip(),
        character_dynamics=character_dynamics_result.strip(),
        world_building=world_building_result.strip()
    )
    plot_arch_result = invoke_with_cleaning(llm_adapter, prompt_plot)

    # 合并
    final_content = (
        "#=== 1) 核心种子 ===\n"
        f"{core_seed_result}\n\n"
        "#=== 2) 角色动力学 ===\n"
        f"{character_dynamics_result}\n\n"
        "#=== 3) 世界观 ===\n"
        f"{world_building_result}\n\n"
        "#=== 4) 三幕式情节架构 ===\n"
        f"{plot_arch_result}\n"
    )

    arch_file = os.path.join(filepath, "Novel_architecture.txt")
    clear_file_content(arch_file)
    save_string_to_txt(final_content, arch_file)
    logging.info("Novel_architecture.txt has been generated successfully.")

# ============ 2) 生成章节蓝图 ============

def Chapter_blueprint_generate(
    api_key: str,
    base_url: str,
    llm_model: str,
    filepath: str,
    temperature: float = 0.7
) -> None:
    arch_file = os.path.join(filepath, "Novel_architecture.txt")
    if not os.path.exists(arch_file):
        logging.warning("Novel_architecture.txt not found. Please generate architecture first.")
        return

    architecture_text = read_file(arch_file).strip()
    if not architecture_text:
        logging.warning("Novel_architecture.txt is empty.")
        return

    match_chaps = re.search(r'约(\d+)章', architecture_text)
    if match_chaps:
        number_of_chapters = int(match_chaps.group(1))
    else:
        number_of_chapters = 10

    # 提取三幕式文本
    plot_arch_text = ""
    pat_plot = r'#=== 4\) 三幕式情节架构 ===\n([\s\S]+)$'
    m = re.search(pat_plot, architecture_text)
    if m:
        plot_arch_text = m.group(1).strip()

    llm_adapter = create_llm_adapter(
        interface_format="openai",  # 或实际由UI传入
        base_url=base_url,
        model_name=llm_model,
        api_key=api_key,
        temperature=temperature
    )

    prompt = chapter_blueprint_prompt.format(
        plot_architecture=plot_arch_text,
        number_of_chapters=number_of_chapters
    )
    blueprint_text = invoke_with_cleaning(llm_adapter, prompt)
    if not blueprint_text.strip():
        logging.warning("Chapter blueprint generation result is empty.")
        return

    filename_dir = os.path.join(filepath, "Novel_directory.txt")
    clear_file_content(filename_dir)
    save_string_to_txt(blueprint_text, filename_dir)

    logging.info("Novel_directory.txt (chapter blueprint) has been generated successfully.")

# ============ 3) 生成章节草稿 ============

def generate_chapter_draft(
    api_key: str,
    base_url: str,
    model_name: str,
    filepath: str,
    novel_number: int,
    word_number: int,
    temperature: float,
    user_guidance: str,
    characters_involved: str,
    key_items: str,
    scene_location: str,
    time_constraint: str,
    embedding_api_key: str,
    embedding_url: str,
    embedding_interface_format: str,
    embedding_model_name: str,
    embedding_retrieval_k: int = 2
) -> str:
    arch_file = os.path.join(filepath, "Novel_architecture.txt")
    novel_architecture_text = read_file(arch_file)

    directory_file = os.path.join(filepath, "Novel_directory.txt")
    blueprint_text = read_file(directory_file)

    global_summary_file = os.path.join(filepath, "global_summary.txt")
    global_summary_text = read_file(global_summary_file)

    character_state_file = os.path.join(filepath, "character_state.txt")
    character_state_text = read_file(character_state_file)

    # 解析本章信息
    chapter_info = get_chapter_info_from_blueprint(blueprint_text, novel_number)
    chapter_title = chapter_info["chapter_title"]
    chapter_role = chapter_info["chapter_role"]
    chapter_purpose = chapter_info["chapter_purpose"]
    suspense_level = chapter_info["suspense_level"]
    foreshadowing = chapter_info["foreshadowing"]
    plot_twist_level = chapter_info["plot_twist_level"]
    chapter_summary = chapter_info["chapter_summary"]

    chapters_dir = os.path.join(filepath, "chapters")
    os.makedirs(chapters_dir, exist_ok=True)

    # 获取最近3章 => (短期摘要, 下一章关键字)
    recent_3_texts = get_last_n_chapters_text(chapters_dir, novel_number, n=3)
    short_summary, next_chapter_keywords = summarize_recent_chapters(
        interface_format="openai",  # 或由UI传进
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        temperature=temperature,
        chapters_text_list=recent_3_texts
    )

    # 上一章片段(末尾1500字)
    previous_chapter_excerpt = ""
    for text_block in reversed(recent_3_texts):
        if text_block.strip():
            if len(text_block) > 1500:
                previous_chapter_excerpt = text_block[-1500:]
            else:
                previous_chapter_excerpt = text_block
            break

    # 使用embedding检索上下文
    embedding_adapter = create_embedding_adapter(
        embedding_interface_format,
        embedding_api_key,
        embedding_url,
        embedding_model_name
    )
    retrieval_query = short_summary + " " + next_chapter_keywords
    relevant_context = get_relevant_context_from_vector_store(
        embedding_adapter=embedding_adapter,
        query=retrieval_query,
        filepath=filepath,
        k=embedding_retrieval_k
    )
    if not relevant_context.strip():
        relevant_context = "（无检索到的上下文）"

    # 组装 Prompt
    prompt_text = chapter_draft_prompt.format(
        novel_number=novel_number,
        chapter_title=chapter_title,
        chapter_role=chapter_role,
        chapter_purpose=chapter_purpose,
        suspense_level=suspense_level,
        foreshadowing=foreshadowing,
        plot_twist_level=plot_twist_level,
        chapter_summary=chapter_summary,

        characters_involved=characters_involved,
        key_items=key_items,
        scene_location=scene_location,
        time_constraint=time_constraint,
        user_guidance=user_guidance,

        novel_setting=novel_architecture_text,
        global_summary=global_summary_text,
        character_state=character_state_text,
        previous_chapter_excerpt=previous_chapter_excerpt,
        context_excerpt=relevant_context
    )

    # 调用 LLM 生成
    llm_adapter = create_llm_adapter(
        interface_format="openai",  # 或由UI传进
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature
    )
    chapter_content = invoke_with_cleaning(llm_adapter, prompt_text)
    if not chapter_content.strip():
        logging.warning("Generated chapter draft is empty.")

    # 写入 chapter_X.txt
    chapter_file = os.path.join(chapters_dir, f"chapter_{novel_number}.txt")
    clear_file_content(chapter_file)
    save_string_to_txt(chapter_content, chapter_file)

    logging.info(f"[Draft] Chapter {novel_number} generated as a draft.")
    return chapter_content

# ============ 4) 定稿章节 ============

def finalize_chapter(
    novel_number: int,
    word_number: int,
    api_key: str,
    base_url: str,
    model_name: str,
    temperature: float,
    filepath: str,
    embedding_api_key: str,
    embedding_url: str,
    embedding_interface_format: str,
    embedding_model_name: str
):
    chapters_dir = os.path.join(filepath, "chapters")
    chapter_file = os.path.join(chapters_dir, f"chapter_{novel_number}.txt")
    chapter_text = read_file(chapter_file).strip()
    if not chapter_text:
        logging.warning(f"Chapter {novel_number} is empty, cannot finalize.")
        return

    # 如果篇幅过短，可以扩写
    if len(chapter_text) < 0.6 * word_number:
        chapter_text = enrich_chapter_text(chapter_text, word_number, api_key, base_url, model_name, temperature)
        clear_file_content(chapter_file)
        save_string_to_txt(chapter_text, chapter_file)

    # 读取全局摘要、角色状态
    global_summary_file = os.path.join(filepath, "global_summary.txt")
    old_global_summary = read_file(global_summary_file)
    character_state_file = os.path.join(filepath, "character_state.txt")
    old_character_state = read_file(character_state_file)

    # 调用 LLM 更新全局摘要
    llm_adapter = create_llm_adapter(
        interface_format="openai",
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature
    )
    prompt_summary = summary_prompt.format(
        chapter_text=chapter_text,
        global_summary=old_global_summary
    )
    new_global_summary = invoke_with_cleaning(llm_adapter, prompt_summary)
    if not new_global_summary.strip():
        new_global_summary = old_global_summary

    # 更新角色状态
    prompt_char_state = update_character_state_prompt.format(
        chapter_text=chapter_text,
        old_state=old_character_state
    )
    new_char_state = invoke_with_cleaning(llm_adapter, prompt_char_state)
    if not new_char_state.strip():
        new_char_state = old_character_state

    # 写回
    clear_file_content(global_summary_file)
    save_string_to_txt(new_global_summary, global_summary_file)

    clear_file_content(character_state_file)
    save_string_to_txt(new_char_state, character_state_file)

    # 更新向量库
    embedding_adapter = create_embedding_adapter(
        embedding_interface_format,
        embedding_api_key,
        embedding_url,
        embedding_model_name
    )
    update_vector_store(embedding_adapter, chapter_text, filepath)

    logging.info(f"Chapter {novel_number} has been finalized.")

def enrich_chapter_text(
    chapter_text: str,
    word_number: int,
    api_key: str,
    base_url: str,
    model_name: str,
    temperature: float
) -> str:
    llm_adapter = create_llm_adapter(
        interface_format="openai",
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature
    )
    prompt = f"""以下章节文本较短，请在保持剧情连贯的前提下进行扩写，使其更充实，接近 {word_number} 字左右：
原内容：
{chapter_text}
"""
    enriched_text = invoke_with_cleaning(llm_adapter, prompt)
    return enriched_text if enriched_text else chapter_text

# ============ 导入知识文件到向量库 ============

def advanced_split_content(content: str,
                           similarity_threshold: float = 0.7,
                           max_length: int = 500) -> List[str]:
    nltk.download('punkt', quiet=True)
    sentences = nltk.sent_tokenize(content)
    if not sentences:
        return []

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(sentences)

    merged_paragraphs = []
    current_sentences = [sentences[0]]
    current_embedding = embeddings[0]

    for i in range(1, len(sentences)):
        sim = cosine_similarity([current_embedding], [embeddings[i]])[0][0]
        if sim >= similarity_threshold:
            current_sentences.append(sentences[i])
            current_embedding = (current_embedding + embeddings[i]) / 2.0
        else:
            merged_paragraphs.append(" ".join(current_sentences))
            current_sentences = [sentences[i]]
            current_embedding = embeddings[i]

    if current_sentences:
        merged_paragraphs.append(" ".join(current_sentences))

    final_segments = []
    for para in merged_paragraphs:
        if len(para) > max_length:
            sub_segments = split_by_length(para, max_length=max_length)
            final_segments.extend(sub_segments)
        else:
            final_segments.append(para)

    return final_segments

def import_knowledge_file(
    embedding_api_key: str,
    embedding_url: str,
    embedding_interface_format: str,
    embedding_model_name: str,
    file_path: str,
    filepath: str
):
    logging.info(f"开始导入知识库文件: {file_path}, 接口格式: {embedding_interface_format}, 模型: {embedding_model_name}")
    if not os.path.exists(file_path):
        logging.warning(f"知识库文件不存在: {file_path}")
        return

    content = read_file(file_path)
    if not content.strip():
        logging.warning("知识库文件内容为空。")
        return

    paragraphs = advanced_split_content(content)

    embedding_adapter = create_embedding_adapter(
        interface_format=embedding_interface_format,
        api_key=embedding_api_key,
        base_url=embedding_url if embedding_url else "http://localhost:11434/api",
        model_name=embedding_model_name
    )

    store = load_vector_store(embedding_adapter, filepath)
    if not store:
        logging.info("Vector store does not exist. Initializing a new one for knowledge import...")
        init_vector_store(embedding_adapter, paragraphs, filepath)
    else:
        docs = [Document(page_content=str(p)) for p in paragraphs]
        store.add_documents(docs)
    logging.info("知识库文件已成功导入至向量库。")
