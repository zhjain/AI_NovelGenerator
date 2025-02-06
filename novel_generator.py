# novel_generator.py
# -*- coding: utf-8 -*-
import os
import logging
import re
import time
import traceback
from typing import List, Optional

# langchain 相关
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
    scene_dynamics_prompt
)

# Ollama嵌入 (如使用Ollama时需要)
from embedding_ollama import OllamaEmbeddings

# 用于目录解析章节标题/简介
from chapter_directory_parser import get_chapter_info_from_blueprint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ============ 基础工具 ============
def remove_think_tags(text: str) -> str:
    """移除 <think>...</think> 包裹的内容"""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def debug_log(prompt: str, response_content: str):
    logging.info(f"\n[#########################################  Prompt  #########################################]\n {prompt}\n")
    logging.info(f"\n[######################################### Response #########################################]\n {response_content}\n")

def invoke_with_cleaning(model: ChatOpenAI, prompt: str) -> str:
    """通用封装：调用模型并移除 <think>...</think> 文本，记录日志后返回"""
    response = model.invoke(prompt)
    if not response:
        logging.warning("No response from model.")
        return ""
    cleaned_text = remove_think_tags(response.content)
    debug_log(prompt, cleaned_text)
    return cleaned_text.strip()

def ensure_openai_base_url_has_v1(url: str) -> str:
    """
    若用户输入的 url 不包含 '/v1'，则在末尾追加 '/v1'。
    """
    import re
    url = url.strip()
    if not url:
        return url
    if not re.search(r'/v\d+$', url):
        if '/v1' not in url:
            url = url.rstrip('/') + '/v1'
    return url

def is_using_ollama_api(interface_format: str) -> bool:
    return interface_format.lower() == "ollama"

def is_using_ml_studio_api(interface_format: str) -> bool:
    return interface_format.lower() == "ml studio"


# ============ 获取 vectorstore 路径 ============
def get_vectorstore_dir(filepath: str) -> str:
    """
    返回存储向量库的本地路径：
    在用户指定的 `filepath` 下创建/使用 'vectorstore' 文件夹。
    """
    return os.path.join(filepath, "vectorstore")


# ============ 创建 Embeddings 对象 ============
def create_embeddings_object(
    api_key: str,
    base_url: str,
    interface_format: str,
    embedding_model_name: str
):
    """
    根据 embedding_interface_format，选择 Ollama 或 OpenAIEmbeddings 等不同后端。
    """
    if is_using_ollama_api(interface_format):
        fixed_url = base_url.rstrip("/")
        return OllamaEmbeddings(
            model_name=embedding_model_name,
            base_url=fixed_url
        )
    else:
        # OpenAI 或 ML Studio 均使用 OpenAIEmbeddings，注意 base_url 可能需要 ensure /v1
        fixed_url = ensure_openai_base_url_has_v1(base_url)
        return OpenAIEmbeddings(
            openai_api_key=api_key,
            openai_api_base=fixed_url,
            model=embedding_model_name
        )


# ============ 向量库相关操作 ============
def clear_vector_store(filepath: str) -> bool:
    """
    返回值表示是否成功清空向量库。
    """
    import shutil

    store_dir = get_vectorstore_dir(filepath)
    if not os.path.exists(store_dir):
        logging.info("No vector store found to clear.")
        return False

    try:
        if os.path.exists(store_dir):
            shutil.rmtree(store_dir)
            logging.info(f"Vector store directory '{store_dir}' removed.")
        return True
    except Exception as e:
        logging.error(f"程序正在运行，无法删除，请在程序关闭后手动前往 {store_dir} 删除目录。\n {str(e)}")
        traceback.print_exc()
        return False

def init_vector_store(
    api_key: str,
    base_url: str,
    interface_format: str,
    embedding_model_name: str,
    texts: List[str],
    filepath: str
) -> Chroma:
    """
    在 filepath 下创建/加载一个 Chroma 向量库并插入 texts。
    """
    store_dir = get_vectorstore_dir(filepath)
    os.makedirs(store_dir, exist_ok=True)

    embeddings = create_embeddings_object(
        api_key=api_key,
        base_url=base_url,
        interface_format=interface_format,
        embedding_model_name=embedding_model_name
    )
    documents = [Document(page_content=str(t)) for t in texts]
    vectorstore = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=store_dir,
        client_settings=Settings(anonymized_telemetry=False),
        collection_name="novel_collection"
    )
    return vectorstore


def load_vector_store(
    api_key: str,
    base_url: str,
    interface_format: str,
    embedding_model_name: str,
    filepath: str
) -> Optional[Chroma]:
    """
    读取已存在的 Chroma 向量库。若不存在则返回 None。
    """
    store_dir = get_vectorstore_dir(filepath)
    if not os.path.exists(store_dir):
        logging.info("Vector store not found. Will return None.")
        return None

    embeddings = create_embeddings_object(
        api_key=api_key,
        base_url=base_url,
        interface_format=interface_format,
        embedding_model_name=embedding_model_name
    )
    return Chroma(
        persist_directory=store_dir,
        embedding_function=embeddings,
        client_settings=Settings(anonymized_telemetry=False),
        collection_name="novel_collection"
    )


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
    """
    if not chapter_text.strip():
        return []

    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    sentences = nltk.sent_tokenize(chapter_text)
    if not sentences:
        return []

    # 先对相近句子进行合并
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

    # 再对合并好的段落做 max_length 切分
    final_segments = []
    for para in merged_paragraphs:
        if len(para) > max_length:
            sub_segments = split_by_length(para, max_length=max_length)
            final_segments.extend(sub_segments)
        else:
            final_segments.append(para)

    return final_segments


def update_vector_store(
    api_key: str,
    base_url: str,
    new_chapter: str,
    interface_format: str,
    embedding_model_name: str,
    filepath: str
):
    """
    将最新章节文本插入到向量库中。若库不存在则初始化。
    """
    splitted_texts = split_text_for_vectorstore(new_chapter)
    if not splitted_texts:
        logging.warning("No valid text to insert into vector store. Skipping.")
        return

    store = load_vector_store(
        api_key=api_key,
        base_url=base_url,
        interface_format=interface_format,
        embedding_model_name=embedding_model_name,
        filepath=filepath
    )
    if not store:
        logging.info("Vector store does not exist. Initializing a new one for new chapter...")
        init_vector_store(
            api_key=api_key,
            base_url=base_url,
            interface_format=interface_format,
            embedding_model_name=embedding_model_name,
            texts=splitted_texts,
            filepath=filepath
        )
        return

    docs = [Document(page_content=str(t)) for t in splitted_texts]
    store.add_documents(docs)
    logging.info("Vector store updated with the new chapter splitted segments.")


def get_relevant_context_from_vector_store(
    api_key: str,
    base_url: str,
    query: str,
    interface_format: str,
    embedding_model_name: str,
    filepath: str,
    k: int = 2
) -> str:
    """
    从向量库中检索与 query 最相关的 k 条文本，拼接后返回。
    """
    store = load_vector_store(
        api_key=api_key,
        base_url=base_url,
        interface_format=interface_format,
        embedding_model_name=embedding_model_name,
        filepath=filepath
    )
    if not store:
        logging.info("No vector store found. Returning empty context.")
        return ""

    docs = store.similarity_search(query, k=k)
    if not docs:
        logging.info(f"No relevant documents found for query '{query}'. Returning empty context.")
        return ""

    combined = "\n".join([d.page_content for d in docs])
    return combined


# ========== 1) 生成总体架构 (Novel_architecture.txt) ==========
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
    依次调用：
      1. core_seed_prompt
      2. character_dynamics_prompt
      3. world_building_prompt
      4. plot_architecture_prompt
    将结果整合为“Novel_architecture.txt”。
    """
    os.makedirs(filepath, exist_ok=True)
    model = ChatOpenAI(
        model=llm_model,
        api_key=api_key,
        base_url=ensure_openai_base_url_has_v1(base_url),
        temperature=temperature
    )

    # 1) 核心种子
    prompt_core = core_seed_prompt.format(
        topic=topic,
        genre=genre,
        number_of_chapters=number_of_chapters,
        word_number=word_number
    )
    core_seed_result = invoke_with_cleaning(model, prompt_core)
    core_seed_text = core_seed_result.strip()

    # 2) 角色动力学
    prompt_character = character_dynamics_prompt.format(core_seed=core_seed_text)
    character_dynamics_result = invoke_with_cleaning(model, prompt_character)
    character_dynamics_text = character_dynamics_result.strip()

    # 3) 世界观
    prompt_world = world_building_prompt.format(core_seed=core_seed_text)
    world_building_result = invoke_with_cleaning(model, prompt_world)
    world_building_text = world_building_result.strip()

    # 4) 三幕式情节架构
    prompt_plot = plot_architecture_prompt.format(
        core_seed=core_seed_text,
        character_dynamics=character_dynamics_text,
        world_building=world_building_text
    )
    plot_arch_result = invoke_with_cleaning(model, prompt_plot)
    plot_arch_text = plot_arch_result.strip()

    # 整合并写入 Novel_architecture.txt
    final_content = (
        "#=== 1) 核心种子 ===\n"
        f"{core_seed_text}\n\n"
        "#=== 2) 角色动力学 ===\n"
        f"{character_dynamics_text}\n\n"
        "#=== 3) 世界观 ===\n"
        f"{world_building_text}\n\n"
        "#=== 4) 三幕式情节架构 ===\n"
        f"{plot_arch_text}\n"
    )

    arch_file = os.path.join(filepath, "Novel_architecture.txt")
    clear_file_content(arch_file)
    save_string_to_txt(final_content, arch_file)

    logging.info("Novel_architecture.txt has been generated successfully.")


# ========== 2) 生成章节蓝图 (Novel_directory.txt) ==========
def Chapter_blueprint_generate(
    api_key: str,
    base_url: str,
    llm_model: str,
    filepath: str,
    temperature: float = 0.7
) -> None:
    """
    基于“Novel_architecture.txt”中的三幕式情节架构，调用 chapter_blueprint_prompt，
    生成章节蓝图并写入 Novel_directory.txt。
    """
    arch_file = os.path.join(filepath, "Novel_architecture.txt")
    if not os.path.exists(arch_file):
        logging.warning("Novel_architecture.txt not found. Please generate architecture first.")
        return

    architecture_text = read_file(arch_file).strip()
    if not architecture_text:
        logging.warning("Novel_architecture.txt is empty.")
        return

    # 从内容中尽量提取 number_of_chapters
    match_chaps = re.search(r'约(\d+)章', architecture_text)
    if match_chaps:
        number_of_chapters = int(match_chaps.group(1))
    else:
        number_of_chapters = 10  # fallback

    # 提取三幕式文本
    plot_arch_text = ""
    pat_plot = r'#=== 4\) 三幕式情节架构 ===\n([\s\S]+)$'
    m = re.search(pat_plot, architecture_text)
    if m:
        plot_arch_text = m.group(1).strip()

    model = ChatOpenAI(
        model=llm_model,
        api_key=api_key,
        base_url=ensure_openai_base_url_has_v1(base_url),
        temperature=temperature
    )

    prompt = chapter_blueprint_prompt.format(
        plot_architecture=plot_arch_text,
        number_of_chapters=number_of_chapters
    )
    blueprint_text = invoke_with_cleaning(model, prompt)
    if not blueprint_text.strip():
        logging.warning("Chapter blueprint generation result is empty.")
        return

    filename_dir = os.path.join(filepath, "Novel_directory.txt")
    clear_file_content(filename_dir)
    save_string_to_txt(blueprint_text, filename_dir)

    logging.info("Novel_directory.txt (chapter blueprint) has been generated successfully.")


# ============ 获取最近 N 章内容，生成短期摘要 ============
def get_last_n_chapters_text(chapters_dir: str, current_chapter_num: int, n: int = 3) -> List[str]:
    texts = []
    start_chap = max(1, current_chapter_num - n)
    for c in range(start_chap, current_chapter_num):
        chap_file = os.path.join(chapters_dir, f"chapter_{c}.txt")
        if os.path.exists(chap_file):
            text = read_file(chap_file).strip()
            if text:
                texts.append(text)
    if len(texts) < n:
        texts = [''] * (n - len(texts)) + texts
    return texts

def summarize_recent_chapters(
    llm_model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    chapters_text_list: List[str]
) -> str:
    if not chapters_text_list:
        return ""
    if all(not txt.strip() for txt in chapters_text_list):
        return "暂无摘要。"

    model = ChatOpenAI(
        model=llm_model,
        api_key=api_key,
        base_url=ensure_openai_base_url_has_v1(base_url),
        temperature=temperature
    )

    combined_text = "\n".join(chapters_text_list)
    prompt = f"""你是一名资深长篇小说编辑，分析以下合并文本：\n\n {combined_text} \n\n

从中提取并预测下一章节的关键字[关键物品/人物/地点/事件/情节]
"""

    summary_text = invoke_with_cleaning(model, prompt)
    if not summary_text:
        return (combined_text[:800] + "...") if len(combined_text) > 800 else combined_text
    return summary_text


# ============ 剧情要点/冲突 ============
PLOT_ARCS_PROMPT = """\
下面是新生成的章节内容:
{chapter_text}

这里是已记录的剧情要点/未解决冲突(可能为空):
{old_plot_arcs}

请基于新的章节内容，提炼本章引入或延续的悬念、冲突、角色暗线等，将其合并到旧的剧情要点中。
若有新的冲突则添加，若有已解决/不再重要的冲突可标注或移除。
最终输出更新后的剧情要点列表，以帮助后续保持故事整体的一致性和悬念延续。
"""

def update_plot_arcs(
    chapter_text: str,
    old_plot_arcs: str,
    api_key: str,
    base_url: str,
    model_name: str,
    temperature: float
) -> str:
    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=ensure_openai_base_url_has_v1(base_url),
        temperature=temperature
    )
    prompt = PLOT_ARCS_PROMPT.format(
        chapter_text=chapter_text,
        old_plot_arcs=old_plot_arcs
    )
    arcs_text = invoke_with_cleaning(model, prompt)
    if not arcs_text:
        logging.warning("update_plot_arcs: No response or empty result.")
        return old_plot_arcs
    return arcs_text


# ========== 3) 生成章节草稿 ==========
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
    """
    根据 scene_dynamics_prompt，生成本章草稿。
    - novel_architecture 取自 Novel_architecture.txt
    - blueprint 取自 Novel_directory.txt
    - global_summary, character_state 分别取自全局摘要、角色状态文件
    - 从向量库检索上下文（embedding_*参数）
    - 用户还可以额外提供四个可选元素：核心人物、关键道具、空间坐标、时间压力
    """

    # 1) 读取相关文件
    arch_file = os.path.join(filepath, "Novel_architecture.txt")
    novel_architecture_text = read_file(arch_file)

    directory_file = os.path.join(filepath, "Novel_directory.txt")
    blueprint_text = read_file(directory_file)

    global_summary_file = os.path.join(filepath, "global_summary.txt")
    global_summary_text = read_file(global_summary_file)

    character_state_file = os.path.join(filepath, "character_state.txt")
    character_state_text = read_file(character_state_file)

    # 2) 解析 blueprint，得到本章所需的字段
    chapter_info = get_chapter_info_from_blueprint(blueprint_text, novel_number)
    chapter_title = chapter_info["chapter_title"]
    chapter_role = chapter_info["chapter_role"]
    chapter_purpose = chapter_info["chapter_purpose"]
    suspense_level = chapter_info["suspense_level"]
    foreshadowing = chapter_info["foreshadowing"]
    plot_twist_level = chapter_info["plot_twist_level"]
    chapter_summary = chapter_info["chapter_summary"]

    # 3) 取最近3章文本，拼成查询语句 => 用于向量库检索
    chapters_dir = os.path.join(filepath, "chapters")
    recent_3_texts = get_last_n_chapters_text(chapters_dir, novel_number, n=3)
    merged_query_str = "回顾剧情：\n" + "\n".join(recent_3_texts) + "\n" + user_guidance

    # 4) 检索向量库上下文 (使用embedding_*参数)
    relevant_context = get_relevant_context_from_vector_store(
        api_key=embedding_api_key,
        base_url=embedding_url,
        query=merged_query_str,
        interface_format=embedding_interface_format,
        embedding_model_name=embedding_model_name,
        filepath=filepath,
        k=embedding_retrieval_k
    )
    if not relevant_context.strip():
        relevant_context = "（无检索到的上下文）"

    novel_setting_text = novel_architecture_text

    prompt_text = scene_dynamics_prompt.format(
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

        novel_setting=novel_setting_text,
        global_summary=global_summary_text,
        character_state=character_state_text
    )

    # 合并检索到的上下文和用户指导
    prompt_text += f"\n\n【检索到的上下文】\n{relevant_context}"
    prompt_text += f"\n\n【章节额外指导】\n{user_guidance}\n"

    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=ensure_openai_base_url_has_v1(base_url),
        temperature=temperature
    )

    chapter_content = invoke_with_cleaning(model, prompt_text)
    if not chapter_content.strip():
        logging.warning("Generated chapter draft is empty.")

    # 6) 写入 chapters
    os.makedirs(chapters_dir, exist_ok=True)
    chapter_file = os.path.join(chapters_dir, f"chapter_{novel_number}.txt")

    clear_file_content(chapter_file)
    save_string_to_txt(chapter_content, chapter_file)

    logging.info(f"[Draft] Chapter {novel_number} generated as a draft.")
    return chapter_content


# ========== 4) 定稿章节 ==========
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
    """
    定稿：更新全局摘要、角色状态，并将本章文本插入向量库。
    """
    chapters_dir = os.path.join(filepath, "chapters")
    chapter_file = os.path.join(chapters_dir, f"chapter_{novel_number}.txt")
    chapter_text = read_file(chapter_file).strip()
    if not chapter_text:
        logging.warning(f"Chapter {novel_number} is empty, cannot finalize.")
        return

    # 若篇幅过短，可尝试扩写
    if len(chapter_text) < 0.6 * word_number:
        chapter_text = enrich_chapter_text(chapter_text, word_number, api_key, base_url, model_name, temperature)
        clear_file_content(chapter_file)
        save_string_to_txt(chapter_text, chapter_file)

    # 读取全局摘要、角色状态
    global_summary_file = os.path.join(filepath, "global_summary.txt")
    old_global_summary = read_file(global_summary_file)
    character_state_file = os.path.join(filepath, "character_state.txt")
    old_character_state = read_file(character_state_file)

    # 1) 更新全局摘要
    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=ensure_openai_base_url_has_v1(base_url),
        temperature=temperature
    )
    prompt_summary = summary_prompt.format(
        chapter_text=chapter_text,
        global_summary=old_global_summary
    )
    new_global_summary = invoke_with_cleaning(model, prompt_summary)
    if not new_global_summary.strip():
        new_global_summary = old_global_summary

    # 2) 更新角色状态
    prompt_char_state = update_character_state_prompt.format(
        chapter_text=chapter_text,
        old_state=old_character_state
    )
    new_char_state = invoke_with_cleaning(model, prompt_char_state)
    if not new_char_state.strip():
        new_char_state = old_character_state

    # 写回文件
    clear_file_content(global_summary_file)
    save_string_to_txt(new_global_summary, global_summary_file)

    clear_file_content(character_state_file)
    save_string_to_txt(new_char_state, character_state_file)

    # 3) 更新向量库 (embedding相关)
    update_vector_store(
        api_key=embedding_api_key,
        base_url=embedding_url,
        new_chapter=chapter_text,
        interface_format=embedding_interface_format,
        embedding_model_name=embedding_model_name,
        filepath=filepath
    )

    logging.info(f"Chapter {novel_number} has been finalized.")


def enrich_chapter_text(
    chapter_text: str,
    word_number: int,
    api_key: str,
    base_url: str,
    model_name: str,
    temperature: float
) -> str:
    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=ensure_openai_base_url_has_v1(base_url),
        temperature=temperature
    )
    prompt = f"""以下是当前章节文本，可能篇幅较短，请在保持剧情连贯的前提下进行扩写，使其更充实、生动，并尽量靠近目标 {word_number} 字数。

原章节内容：
{chapter_text}"""
    enriched_text = invoke_with_cleaning(model, prompt)
    return enriched_text if enriched_text else chapter_text


# ============ 导入外部知识文本到向量库 ============
def advanced_split_content(content: str,
                           similarity_threshold: float = 0.7,
                           max_length: int = 500) -> List[str]:
    """
    将文本先按句子切分，然后根据语义相似度进行合并，最后按 max_length 二次切分。
    """
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

    # 尝试加载已有的向量库
    store = load_vector_store(
        api_key=embedding_api_key,
        base_url=embedding_url if embedding_url else "http://localhost:11434/api",
        interface_format=embedding_interface_format,
        embedding_model_name=embedding_model_name,
        filepath=filepath
    )
    if not store:
        logging.info("Vector store does not exist. Initializing a new one for knowledge import...")
        init_vector_store(
            api_key=embedding_api_key,
            base_url=embedding_url if embedding_url else "http://localhost:11434/api",
            interface_format=embedding_interface_format,
            embedding_model_name=embedding_model_name,
            texts=paragraphs,
            filepath=filepath
        )
    else:
        docs = [Document(page_content=str(p)) for p in paragraphs]
        store.add_documents(docs)
    logging.info("知识库文件已成功导入至向量库。")
