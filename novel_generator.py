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
import math
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 工具函数
from utils import (
    read_file, append_text_to_file, clear_file_content,
    save_string_to_txt
)

# prompt模板
from prompt_definitions import (
    # 设定相关
    set_prompt, character_prompt, dark_lines_prompt,
    finalize_setting_prompt, novel_directory_prompt,

    # 写作流程相关
    summary_prompt, update_character_state_prompt,
    chapter_outline_prompt, chapter_write_prompt
)

# Ollama嵌入 (如使用Ollama时需要)
from embedding_ollama import OllamaEmbeddings

# 用于目录解析章节标题/简介
from chapter_directory_parser import get_chapter_info_from_directory


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ============ 帮助函数 ============
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


# ============ 1. 生成小说“设定” (Novel_setting.txt) ============
def Novel_setting_generate(
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
    os.makedirs(filepath, exist_ok=True)

    model = ChatOpenAI(
        model=llm_model,
        api_key=api_key,
        base_url=ensure_openai_base_url_has_v1(base_url),
        temperature=temperature
    )

    # Step1: 基础设定
    prompt_base = set_prompt.format(
        topic=topic,
        genre=genre,
        number_of_chapters=number_of_chapters,
        word_number=word_number
    )
    base_setting = invoke_with_cleaning(model, prompt_base)

    # Step2: 角色设定
    prompt_char = character_prompt.format(
        novel_setting=base_setting
    )
    character_setting = invoke_with_cleaning(model, prompt_char)

    # Step3: 暗线/雷点
    prompt_dark = dark_lines_prompt.format(
        character_info=character_setting
    )
    dark_lines = invoke_with_cleaning(model, prompt_dark)

    # Step4: 最终整合
    prompt_final = finalize_setting_prompt.format(
        novel_setting_base=base_setting,
        character_setting=character_setting,
        dark_lines=dark_lines
    )
    final_novel_setting = invoke_with_cleaning(model, prompt_final)

    filename_set = os.path.join(filepath, "Novel_setting.txt")
    clear_file_content(filename_set)

    final_novel_setting_cleaned = final_novel_setting.replace('#', '').replace('*', '')
    save_string_to_txt(final_novel_setting_cleaned, filename_set)
    logging.info("Novel_setting.txt has been generated successfully.")


# ============ 2. 生成小说目录 (Novel_directory.txt) ============
def Novel_directory_generate(
    api_key: str,
    base_url: str,
    llm_model: str,
    number_of_chapters: int,
    filepath: str,
    temperature: float = 0.7
) -> None:
    filename_set = os.path.join(filepath, "Novel_setting.txt")
    final_novel_setting = read_file(filename_set).strip()
    if not final_novel_setting:
        logging.warning("Novel_setting.txt 内容为空，请先生成小说设定。")
        return

    model = ChatOpenAI(
        model=llm_model,
        api_key=api_key,
        base_url=ensure_openai_base_url_has_v1(base_url),
        temperature=temperature
    )

    prompt_dir = novel_directory_prompt.format(
        final_novel_setting=final_novel_setting,
        number_of_chapters=number_of_chapters
    )
    final_novel_directory = invoke_with_cleaning(model, prompt_dir)
    if not final_novel_directory.strip():
        logging.warning("Novel_directory生成结果为空。")
        return

    filename_dir = os.path.join(filepath, "Novel_directory.txt")
    clear_file_content(filename_dir)

    final_novel_directory_cleaned = final_novel_directory.replace('#', '').replace('*', '')
    save_string_to_txt(final_novel_directory_cleaned, filename_dir)

    logging.info("Novel_directory.txt has been generated successfully.")


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
    prompt = f"""你是一名资深长篇小说写作辅助AI，下面是最近几章的合并文本：
{combined_text}

请用中文输出不超过500字的摘要，只包含主要剧情进展、角色变化、冲突焦点等要点："""

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


# ============ 生成章节草稿 ============
def generate_chapter_draft(
    novel_settings: str,
    global_summary: str,
    character_state: str,
    recent_chapters_summary: str,
    user_guidance: str,
    api_key: str,
    base_url: str,
    model_name: str,
    novel_number: int,
    word_number: int,
    temperature: float,
    novel_novel_directory: str,
    filepath: str,
    interface_format: str,
    embedding_model_name: str,
    embedding_base_url: str,
    embedding_retrieval_k: int = 4
) -> str:
    # 1) 根据目录解析标题、简介
    chapter_info = get_chapter_info_from_directory(novel_novel_directory, novel_number)
    chapter_title = chapter_info["chapter_title"]
    chapter_brief = chapter_info["chapter_brief"]

    # 合并要检索的文本（用户指导 + 章节简介 + 最近摘要）
    combined_query_parts = []
    if user_guidance.strip():
        combined_query_parts.append(user_guidance)
    if chapter_brief.strip():
        combined_query_parts.append(chapter_brief)
    if recent_chapters_summary.strip():
        combined_query_parts.append(recent_chapters_summary)
    # 额外加一个关键字
    combined_query_parts.append("回顾剧情")

    merged_query_str = "\n".join(combined_query_parts)

    # 2) 从向量库检索上下文
    relevant_context = get_relevant_context_from_vector_store(
        api_key=api_key,
        base_url=embedding_base_url if embedding_base_url else base_url,
        query=merged_query_str,
        interface_format=interface_format,
        embedding_model_name=embedding_model_name,
        filepath=filepath,
        k=embedding_retrieval_k
    )
    if not relevant_context.strip():
        relevant_context = "暂无相关内容。"

    # 3) 生成本章大纲
    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=ensure_openai_base_url_has_v1(base_url),
        temperature=temperature
    )

    outline_prompt_text = chapter_outline_prompt.format(
        novel_setting=novel_settings,
        character_state=character_state + "\n\n【检索到的上下文】\n" + relevant_context,
        global_summary=global_summary,
        novel_number=novel_number,
        chapter_title=chapter_title,
        chapter_brief=chapter_brief
    )
    outline_prompt_text += f"\n\n【最近几章摘要】\n{recent_chapters_summary}"
    outline_prompt_text += f"\n\n【用户指导】\n{user_guidance if user_guidance else '（无）'}"

    chapter_outline = invoke_with_cleaning(model, outline_prompt_text)

    outlines_dir = os.path.join(filepath, "outlines")
    os.makedirs(outlines_dir, exist_ok=True)
    outline_file = os.path.join(outlines_dir, f"outline_{novel_number}.txt")
    clear_file_content(outline_file)
    save_string_to_txt(chapter_outline, outline_file)

    # 4) 生成正文草稿
    writing_prompt_text = chapter_write_prompt.format(
        novel_setting=novel_settings,
        character_state=character_state + "\n\n【检索到的上下文】\n" + relevant_context,
        global_summary=global_summary,
        chapter_outline=chapter_outline,
        word_number=word_number,
        chapter_title=chapter_title,
        chapter_brief=chapter_brief
    )
    writing_prompt_text += f"\n\n【最近几章摘要】\n{recent_chapters_summary}"
    writing_prompt_text += f"\n\n【用户指导】\n{user_guidance if user_guidance else '（无）'}"

    chapter_content = invoke_with_cleaning(model, writing_prompt_text)

    chapters_dir = os.path.join(filepath, "chapters")
    os.makedirs(chapters_dir, exist_ok=True)
    chapter_file = os.path.join(chapters_dir, f"chapter_{novel_number}.txt")
    clear_file_content(chapter_file)
    save_string_to_txt(chapter_content, chapter_file)

    logging.info(f"[Draft] Chapter {novel_number} generated as a draft.")
    return chapter_content


# ============ 定稿章节 ============
def finalize_chapter(
    novel_number: int,
    word_number: int,
    api_key: str,
    base_url: str,
    interface_format: str,
    embedding_model_name: str,
    model_name: str,
    temperature: float,
    filepath: str,
    embedding_base_url: str,
    embedding_api_key: str
):
    chapters_dir = os.path.join(filepath, "chapters")
    chapter_file = os.path.join(chapters_dir, f"chapter_{novel_number}.txt")
    chapter_text = read_file(chapter_file).strip()
    if not chapter_text:
        logging.warning(f"Chapter {novel_number} is empty, cannot finalize.")
        return

    character_state_file = os.path.join(filepath, "character_state.txt")
    global_summary_file = os.path.join(filepath, "global_summary.txt")
    plot_arcs_file = os.path.join(filepath, "plot_arcs.txt")

    old_char_state = read_file(character_state_file)
    old_global_summary = read_file(global_summary_file)
    old_plot_arcs = read_file(plot_arcs_file)

    # 篇幅不足，二次扩写
    if len(chapter_text) < 0.8 * word_number:
        logging.info("Chapter text is shorter than 80% of desired length. Enriching...")
        chapter_text = enrich_chapter_text(
            chapter_text=chapter_text,
            word_number=word_number,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            temperature=temperature
        )
        clear_file_content(chapter_file)
        save_string_to_txt(chapter_text, chapter_file)

    # 更新全局摘要
    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=ensure_openai_base_url_has_v1(base_url),
        temperature=temperature
    )

    def update_global_summary(chapter_text: str, old_summary: str) -> str:
        prompt = summary_prompt.format(
            chapter_text=chapter_text,
            global_summary=old_summary
        )
        return invoke_with_cleaning(model, prompt) or old_summary

    new_global_summary = update_global_summary(chapter_text, old_global_summary)

    # 更新角色状态
    def update_character_state(chapter_text: str, old_state: str) -> str:
        prompt = update_character_state_prompt.format(
            chapter_text=chapter_text,
            old_state=old_state
        )
        return invoke_with_cleaning(model, prompt) or old_state

    new_char_state = update_character_state(chapter_text, old_char_state)

    # 更新剧情要点
    new_plot_arcs = update_plot_arcs(
        chapter_text=chapter_text,
        old_plot_arcs=old_plot_arcs,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        temperature=temperature
    )

    # 写回文件
    clear_file_content(character_state_file)
    save_string_to_txt(new_char_state, character_state_file)

    clear_file_content(global_summary_file)
    save_string_to_txt(new_global_summary, global_summary_file)

    clear_file_content(plot_arcs_file)
    save_string_to_txt(new_plot_arcs, plot_arcs_file)

    # 更新向量库（此时用 embedding_api_key/embedding_base_url）
    update_vector_store(
        api_key=embedding_api_key,
        base_url=embedding_base_url if embedding_base_url else base_url,
        new_chapter=chapter_text,
        interface_format=interface_format,
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
    api_key: str,
    base_url: str,
    interface_format: str,
    embedding_model_name: str,
    file_path: str,
    embedding_base_url: str,
    filepath: str
):
    logging.info(f"开始导入知识库文件: {file_path}, 接口格式: {interface_format}, 模型: {embedding_model_name}")
    if not os.path.exists(file_path):
        logging.warning(f"知识库文件不存在: {file_path}")
        return

    content = read_file(file_path)
    if not content.strip():
        logging.warning("知识库文件内容为空。")
        return

    paragraphs = advanced_split_content(content)

    # 若向量库不存在则初始化，否则追加
    store = load_vector_store(
        api_key=api_key,
        base_url=base_url if base_url else "http://localhost:11434/v1",
        interface_format=interface_format,
        embedding_model_name=embedding_model_name,
        filepath=filepath
    )
    if not store:
        logging.info("Vector store does not exist. Initializing a new one for knowledge import...")
        init_vector_store(
            api_key=api_key,
            base_url=base_url if base_url else "http://localhost:11434/v1",
            interface_format=interface_format,
            embedding_model_name=embedding_model_name,
            texts=paragraphs,
            filepath=filepath
        )
    else:
        docs = [Document(page_content=str(p)) for p in paragraphs]
        store.add_documents(docs)
    logging.info("知识库文件已成功导入至向量库。")
