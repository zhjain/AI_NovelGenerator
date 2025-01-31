# novel_generator.py
# -*- coding: utf-8 -*-
import os
import logging
import re
from typing import Dict, List, Optional
try:
    from typing import TypedDict  # Python 3.8+ 直接可用；若是3.7可改用 typing_extensions
except ImportError:
    from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

import nltk
import math
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils import (
    read_file, append_text_to_file, clear_file_content,
    save_string_to_txt
)
from prompt_definitions import (
    set_prompt, character_prompt, dark_lines_prompt,
    finalize_setting_prompt, novel_directory_prompt,
    summary_prompt, update_character_state_prompt,
    chapter_outline_prompt, chapter_write_prompt
)

# ============ 新增：导入 chapter_directory_parser ============
from chapter_directory_parser import get_chapter_info_from_directory

# ============ 日志配置 ============
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
def debug_log(prompt: str, response_content: str):
        """在控制台打印或记录下每次Prompt与Response，[调试]"""
        logging.info(f"\n[Prompt >>>] {prompt}\n")
        logging.info(f"[Response >>>] {response_content}\n")
# ============ 向量检索相关 ============

VECTOR_STORE_DIR = "vectorstore"

def clear_vector_store():
    """
    清空本地向量库（删除 vectorstore 文件夹）。
    需要在UI中加一个二次确认弹窗，防止误删。
    """
    if os.path.exists(VECTOR_STORE_DIR):
        try:
            import shutil
            shutil.rmtree(VECTOR_STORE_DIR)
            logging.info("Local vector store has been cleared.")
        except Exception as e:
            logging.warning(f"Failed to remove vector store: {e}")
    else:
        logging.info("No vector store found to clear.")

def init_vector_store(api_key: str, base_url: str, texts: List[str]) -> Chroma:
    """
    初始化并返回一个Chroma向量库，将传入的文本进行嵌入并保存到本地目录。
    如果不存在该目录，会自动创建。
    """
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        openai_api_base=base_url
    )
    documents = [Document(page_content=t) for t in texts]
    vectorstore = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR
    )
    vectorstore.persist()
    return vectorstore

def load_vector_store(api_key: str, base_url: str) -> Optional[Chroma]:
    """读取已存在的向量库。若不存在则返回 None。"""
    if not os.path.exists(VECTOR_STORE_DIR):
        return None
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        openai_api_base=base_url
    )
    return Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings)

def update_vector_store(api_key: str, base_url: str, new_chapter: str) -> None:
    """将最新章节文本插入到向量库里，用于后续检索参考。若库不存在则初始化。"""
    store = load_vector_store(api_key, base_url)
    if not store:
        logging.info("Vector store does not exist. Initializing a new one...")
        init_vector_store(api_key, base_url, [new_chapter])
        return

    new_doc = Document(page_content=new_chapter)
    store.add_documents([new_doc])
    store.persist()

def get_relevant_context_from_vector_store(api_key: str, base_url: str, query: str, k: int = 2) -> str:
    """
    从向量库中检索与 query 最相关的 k 条文本，拼接后返回。
    若向量库不存在则返回空字符串。
    """
    store = load_vector_store(api_key, base_url)
    if not store:
        logging.warning("Vector store not found. Returning empty context.")
        return ""
    docs = store.similarity_search(query, k=k)
    combined = "\n".join([d.page_content for d in docs])
    return combined

# ============ 多步生成：设置 & 目录 ============

class OverallState(TypedDict):
    topic: str
    genre: str
    number_of_chapters: int
    word_number: int
    novel_setting_base: str
    character_setting: str
    dark_lines: str
    final_novel_setting: str
    novel_directory: str

def Novel_novel_directory_generate(
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
    使用多步流程，生成 Novel_setting.txt 与 Novel_directory.txt 并保存到 filepath。
    """
    # 确保文件夹存在
    os.makedirs(filepath, exist_ok=True)

    model = ChatOpenAI(
        model=llm_model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature
    )

    

    def generate_base_setting(state: OverallState) -> Dict[str, str]:
        prompt = set_prompt.format(
            topic=state["topic"],
            genre=state["genre"],
            number_of_chapters=state["number_of_chapters"],
            word_number=state["word_number"]
        )
        response = model.invoke(prompt)
        if not response:
            logging.warning("generate_base_setting: No response.")
            return {"novel_setting_base": ""}
        debug_log(prompt, response.content)
        return {"novel_setting_base": response.content.strip()}

    def generate_character_setting(state: OverallState) -> Dict[str, str]:
        prompt = character_prompt.format(
            novel_setting=state["novel_setting_base"]
        )
        response = model.invoke(prompt)
        if not response:
            logging.warning("generate_character_setting: No response.")
            return {"character_setting": ""}
        debug_log(prompt, response.content)
        return {"character_setting": response.content.strip()}

    def generate_dark_lines(state: OverallState) -> Dict[str, str]:
        prompt = dark_lines_prompt.format(
            character_info=state["character_setting"]
        )
        response = model.invoke(prompt)
        if not response:
            logging.warning("generate_dark_lines: No response.")
            return {"dark_lines": ""}
        debug_log(prompt, response.content)
        return {"dark_lines": response.content.strip()}

    def finalize_novel_setting(state: OverallState) -> Dict[str, str]:
        prompt = finalize_setting_prompt.format(
            novel_setting_base=state["novel_setting_base"],
            character_setting=state["character_setting"],
            dark_lines=state["dark_lines"]
        )
        response = model.invoke(prompt)
        if not response:
            logging.warning("finalize_novel_setting: No response.")
            return {"final_novel_setting": ""}
        debug_log(prompt, response.content)
        return {"final_novel_setting": response.content.strip()}

    def generate_novel_directory(state: OverallState) -> Dict[str, str]:
        prompt = novel_directory_prompt.format(
            final_novel_setting=state["final_novel_setting"],
            number_of_chapters=state["number_of_chapters"]
        )
        response = model.invoke(prompt)
        if not response:
            logging.warning("generate_novel_directory: No response.")
            return {"novel_directory": ""}
        debug_log(prompt, response.content)
        return {"novel_directory": response.content.strip()}

    # 构建状态图
    graph = StateGraph(OverallState)
    graph.add_node("generate_base_setting", generate_base_setting)
    graph.add_node("generate_character_setting", generate_character_setting)
    graph.add_node("generate_dark_lines", generate_dark_lines)
    graph.add_node("finalize_novel_setting", finalize_novel_setting)
    graph.add_node("generate_novel_directory", generate_novel_directory)

    graph.add_edge(START, "generate_base_setting")
    graph.add_edge("generate_base_setting", "generate_character_setting")
    graph.add_edge("generate_character_setting", "generate_dark_lines")
    graph.add_edge("generate_dark_lines", "finalize_novel_setting")
    graph.add_edge("finalize_novel_setting", "generate_novel_directory")
    graph.add_edge("generate_novel_directory", END)

    app = graph.compile()

    input_params = {
        "topic": topic,
        "genre": genre,
        "number_of_chapters": number_of_chapters,
        "word_number": word_number
    }
    result = app.invoke(input_params)

    if not result:
        logging.warning("Novel_novel_directory_generate: invoke() 结果为空，生成失败。")
        return

    final_novel_setting = result.get("final_novel_setting", "")
    final_novel_directory = result.get("novel_directory", "")

    if not final_novel_setting or not final_novel_directory:
        logging.warning("生成失败：缺少 final_novel_setting 或 novel_directory。")
        return

    # 写入文件
    filename_set = os.path.join(filepath, "Novel_setting.txt")
    filename_novel_directory = os.path.join(filepath, "Novel_directory.txt")

    # 清理文本（可根据需要去除多余字符）
    def clean_text(txt: str) -> str:
        return txt.replace('#', '').replace('*', '')

    final_novel_setting_cleaned = clean_text(final_novel_setting)
    final_novel_directory_cleaned = clean_text(final_novel_directory)

    append_text_to_file(final_novel_setting_cleaned, filename_set)
    append_text_to_file(final_novel_directory_cleaned, filename_novel_directory)

    logging.info("Novel settings and directory generated successfully.")

# ============ 新增：获取最近N章内容，生成短期摘要 ============

def get_last_n_chapters_text(chapters_dir: str, current_chapter_num: int, n: int = 3) -> List[str]:
    """
    从指定文件夹中，读取最近 n 章的内容（如果存在），并按从旧到新的顺序返回文本列表。
    不包含当前章，只拿之前的 n 章。
    """
    texts = []
    start_chap = max(1, current_chapter_num - n)
    for c in range(start_chap, current_chapter_num):
        chap_file = os.path.join(chapters_dir, f"chapter_{c}.txt")
        if os.path.exists(chap_file):
            text = read_file(chap_file).strip()
            if text:
                texts.append(text)
    return texts

def summarize_recent_chapters(model: ChatOpenAI, chapters_text_list: List[str]) -> str:
    """
    将最近几章的文本拼接后，通过模型生成一个相对详细的“短期内容摘要”。
    """
    if not chapters_text_list:
        return ""

    # 拼接这几章的内容
    combined_text = "\n".join(chapters_text_list)
    # 在这里可以写一个更详细的提示
    prompt = f"""\
这是最近几章的故事内容，请生成一份详细的短期内容摘要（不少于一章篇幅的细节），用于帮助后续创作时回顾细节。
请着重强调发生的事件、角色的心理和关系变化、冲突或悬念等。

{combined_text}
"""
    response = model.invoke(prompt)
    if not response:
        return ""
    debug_log(prompt, response.content)
    return response.content.strip()

# ============ 生成章节草稿 & 定稿 ============

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
    filepath: str
) -> str:
    """
    仅生成当前章节的草稿，不更新全局摘要/角色状态/向量库。
    并将生成的内容写到 "chapter_{novel_number}.txt" 覆盖写入。
    同时生成 "outline_{novel_number}.txt" 存储大纲内容。
    recent_chapters_summary: 最近 3 章的“短期内容摘要”
    """

    # 0) 根据 novel_number 从 novel_novel_directory 中获取本章标题及简述
    chapter_info = get_chapter_info_from_directory(novel_novel_directory, novel_number)
    chapter_title = chapter_info["chapter_title"]
    chapter_brief = chapter_info["chapter_brief"]

    # 1) 从向量库检索往期上下文
    relevant_context = get_relevant_context_from_vector_store(
        api_key, base_url, "回顾剧情", k=2
    )

    # 2) 生成大纲
    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature
    )

    # Prompt 拼接
    outline_prompt_text = chapter_outline_prompt.format(
        novel_setting=novel_settings,
        character_state=character_state + "\n\n【历史上下文】\n" + relevant_context,
        global_summary=global_summary,
        novel_number=novel_number,
        chapter_title=chapter_title,
        chapter_brief=chapter_brief
    )

    # 在后面加上用户指导与最近章节摘要（可根据需要灵活组织）
    outline_prompt_text += f"\n\n【本章目录标题与简述】\n标题：{chapter_title}\n简述：{chapter_brief}\n"
    outline_prompt_text += f"\n【最近几章摘要】\n{recent_chapters_summary}"
    outline_prompt_text += f"\n\n【用户指导】\n{user_guidance if user_guidance else '（无）'}"

    response_outline = model.invoke(outline_prompt_text)
    if not response_outline:
        logging.warning("generate_chapter_draft: outline no response.")
        chapter_outline = ""
    else:
        debug_log(outline_prompt_text, response_outline.content)
        chapter_outline = response_outline.content.strip()

    # 将大纲写到 outline_{novel_number}.txt
    outlines_dir = os.path.join(filepath, "outlines")
    os.makedirs(outlines_dir, exist_ok=True)
    outline_file = os.path.join(outlines_dir, f"outline_{novel_number}.txt")
    clear_file_content(outline_file)
    save_string_to_txt(chapter_outline, outline_file)

    # 3) 生成正文草稿
    writing_prompt_text = chapter_write_prompt.format(
        novel_setting=novel_settings,
        character_state=character_state + "\n\n【历史上下文】\n" + relevant_context,
        global_summary=global_summary,
        chapter_outline=chapter_outline,
        word_number=word_number,
        chapter_title=chapter_title,
        chapter_brief=chapter_brief
    )

    # 同样插入用户指导和最近摘要
    writing_prompt_text += f"\n\n【本章目录标题与简述】\n标题：{chapter_title}\n简述：{chapter_brief}\n"
    writing_prompt_text += f"\n【最近几章摘要】\n{recent_chapters_summary}"
    writing_prompt_text += f"\n\n【用户指导】\n{user_guidance if user_guidance else '（无）'}"

    response_chapter = model.invoke(writing_prompt_text)
    if not response_chapter:
        logging.warning("generate_chapter_draft: writing no response.")
        chapter_content = ""
    else:
        debug_log(writing_prompt_text, response_chapter.content)
        chapter_content = response_chapter.content.strip()

    # 4) 覆盖写到 chapter_{novel_number}.txt
    chapters_dir = os.path.join(filepath, "chapters")
    os.makedirs(chapters_dir, exist_ok=True)
    chapter_file = os.path.join(chapters_dir, f"chapter_{novel_number}.txt")
    clear_file_content(chapter_file)
    save_string_to_txt(chapter_content, chapter_file)

    logging.info(f"[Draft] Chapter {novel_number} generated as a draft.")
    return chapter_content

def finalize_chapter(
    novel_number: int,
    word_number: int,
    api_key: str,
    base_url: str,
    model_name: str,
    temperature: float,
    filepath: str
):
    """
    对当前章节进行定稿：
    1. 读取 chapter_{novel_number}.txt 的最终内容；
    2. 更新全局摘要、角色状态文件；
    3. 如果字数明显少于 word_number 的 80%，则自动调用 enrich_chapter_text 再次扩写；
    4. 更新向量库。
    """
    # 读取当前章节内容
    chapters_dir = os.path.join(filepath, "chapters")
    chapter_file = os.path.join(chapters_dir, f"chapter_{novel_number}.txt")
    chapter_text = read_file(chapter_file).strip()
    if not chapter_text:
        logging.warning(f"Chapter {novel_number} is empty, cannot finalize.")
        return

    # 读取角色状态 & 全局摘要
    character_state_file = os.path.join(filepath, "character_state.txt")
    global_summary_file = os.path.join(filepath, "global_summary.txt")

    old_char_state = read_file(character_state_file)
    old_global_summary = read_file(global_summary_file)

    # 1) 先检查字数是否过少，若少于 80% 则调用 enrich 逻辑
    if len(chapter_text) < 0.8 * word_number:
        logging.info("Chapter text seems shorter than 80% of desired length. Attempting to enrich content...")
        chapter_text = enrich_chapter_text(
            chapter_text=chapter_text,
            word_number=word_number,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            temperature=temperature
        )
        # 覆盖写回文件
        clear_file_content(chapter_file)
        save_string_to_txt(chapter_text, chapter_file)
        logging.info("Chapter text has been enriched and updated.")

    # 2) 更新全局摘要
    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature
    )

    def update_global_summary(chapter_text: str, old_summary: str) -> str:
        prompt = summary_prompt.format(
            chapter_text=chapter_text,
            global_summary=old_summary
        )
        response = model.invoke(prompt)
        if not response:
            logging.warning("update_global_summary: No response.")
            return old_summary
        debug_log(prompt, response.content)
        return response.content.strip()

    new_global_summary = update_global_summary(chapter_text, old_global_summary)

    # 3) 更新角色状态
    def update_character_state(chapter_text: str, old_state: str) -> str:
        prompt = update_character_state_prompt.format(
            chapter_text=chapter_text,
            old_state=old_state
        )
        response = model.invoke(prompt)
        if not response:
            logging.warning("update_character_state: No response.")
            return old_state
        debug_log(prompt, response.content)
        return response.content.strip()

    new_char_state = update_character_state(chapter_text, old_char_state)

    # 4) 覆盖写入角色状态文件与全局摘要文件
    clear_file_content(character_state_file)
    save_string_to_txt(new_char_state, character_state_file)

    clear_file_content(global_summary_file)
    save_string_to_txt(new_global_summary, global_summary_file)

    # 5) 更新向量检索库
    update_vector_store(api_key, base_url, chapter_text)

    logging.info(f"Chapter {novel_number} has been finalized (summary & state updated, vector store updated).")

def enrich_chapter_text(
    chapter_text: str,
    word_number: int,
    api_key: str,
    base_url: str,
    model_name: str,
    temperature: float
) -> str:
    """
    当章节篇幅不足时，调用此函数对章节文本进行二次扩写。
    可以让模型补充场景描写、角色心理等，保证与现有文本风格一致。
    """
    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature
    )
    prompt = f"""\
以下是当前章节文本，可能篇幅较短，请在保持剧情连贯的前提下进行扩写，使其更充实、生动，并尽量靠近目标 {word_number} 字数。

原章节内容：
{chapter_text}
"""
    response = model.invoke(prompt)
    if not response:
        logging.warning("enrich_chapter_text: No response.")
        return chapter_text  # 无响应时就返回原文
    debug_log(prompt, response.content)
    return response.content.strip()

# ============ 导入外部知识文本 ============

def import_knowledge_file(api_key: str, base_url: str, file_path: str) -> None:
    """
    将用户选定的文本文件导入到向量库，以便在写作时检索。
    """

    # 1. 检查文件路径是否有效
    if not os.path.exists(file_path):
        logging.warning(f"知识库文件不存在: {file_path}")
        return

    # 2. 读取文件内容
    content = read_file(file_path)
    if not content.strip():
        logging.warning("知识库文件内容为空。")
        return

    # 3. 对内容进行高级切分处理
    paragraphs = advanced_split_content(content)

    # 4. 加载或初始化向量存储
    store = load_vector_store(api_key, base_url)
    if not store:
        logging.info("Vector store does not exist. Initializing a new one for knowledge import...")
        init_vector_store(api_key, base_url, paragraphs)
        return

    # 5. 创建Document对象并更新到向量库
    docs = [Document(page_content=p) for p in paragraphs]
    store.add_documents(docs)
    store.persist()
    logging.info("知识库文件已成功导入至向量库。")

def advanced_split_content(content: str,
                           similarity_threshold: float = 0.7,
                           max_length: int = 500) -> List[str]:
    """
    将文本先按句子切分，然后根据语义相似度进行合并，最后根据max_length进行二次切分。
    """
    nltk.download('punkt', quiet=True)  # 确保 punkt 数据可用
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

    # 按最大长度二次拆分
    final_segments = []
    for para in merged_paragraphs:
        if len(para) > max_length:
            sub_segments = split_by_length(para, max_length=max_length)
            final_segments.extend(sub_segments)
        else:
            final_segments.append(para)

    return final_segments

def split_by_length(text: str, max_length: int = 500) -> List[str]:
    segments = []
    start_idx = 0
    while start_idx < len(text):
        end_idx = min(start_idx + max_length, len(text))
        segment = text[start_idx:end_idx]
        segments.append(segment.strip())
        start_idx = end_idx
    return segments
