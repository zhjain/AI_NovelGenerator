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

# 
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

# ============ 日志配置 ============
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ============ 向量检索相关函数（Chroma） ============

VECTOR_STORE_DIR = "vectorstore"

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

    :param api_key:       OpenAI API key
    :param base_url:      OpenAI API base url
    :param llm_model:     所使用的 LLM 模型名称
    :param topic:         小说主题
    :param genre:         小说类型
    :param number_of_chapters: 章节数
    :param word_number:   单章目标字数
    :param filepath:      存放生成文件的目录路径
    :param temperature:   生成温度
    """
    # 确保文件夹存在
    os.makedirs(filepath, exist_ok=True)

    model = ChatOpenAI(
        model=llm_model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature
    )

    def debug_log(prompt: str, response_content: str):
        """在控制台打印或记录下每次Prompt与Response，[调试]"""
        logging.info(f"\n[Prompt >>>] {prompt}\n")
        logging.info(f"[Response <<<] {response_content}\n")

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

# ============ 生成章节（每章独立文件） ============

CHINESE_NUM_MAP = {
    '零': 0, '○': 0, '〇': 0,
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9,
    '十': 10, '百': 100, '千': 1000, '万': 10000
}

def chinese_to_arabic(chinese_str: str) -> int:
    """
    只能处理到万(10000)以内的中文数字，正常小说章节应该够用了
    """
    total = 0
    current_unit = 1  # 记录当前单位
    tmp_val = 0       # 暂存本轮数字

    for char in reversed(chinese_str):
        if char in CHINESE_NUM_MAP:
            val = CHINESE_NUM_MAP[char]
            if val >= 10:
                if val > current_unit:
                    # 如 100, 1000, 10000
                    current_unit = val
                else:
                    # 比如 “十二” -> 2 * 10 + 1
                    # 如果 val <= current_unit, 那么相当于在这个单位下加
                    total += tmp_val * val
                    tmp_val = 0
            else:
                # 0~9
                tmp_val = tmp_val + val * current_unit
        else:
            # 非中文数字字符，视情况决定怎么处理，这里直接跳过
            pass
    
    total += tmp_val
    return total

def parse_chapter_title_from_directory(novel_directory_text: str, 
                                       novel_number: int, 
                                       range_size: int = 1) -> str:
    """
    从小说目录文本中，提取指定章节（以及前后几章）的目录信息。
    range_size=1，表示获取当前章节、前一章和后一章的目录信息(若存在)。
    支持多种常见的章节格式。
    """

    lines = novel_directory_text.splitlines()

    # 可以根据需求自行扩展，这里列举了几种常见的章节标题格式，如果模型实在不听话，可以适当调整
    # 每个pattern都应该捕获两个组：
    # 1. chapter_num_str：章节数字(可能是中文也可能是阿拉伯数字)
    # 2. chapter_title ：章节标题（.*）
    patterns = [
        # 1) 第12章 标题
        r"^第\s*([\d]+)\s*章[:：]?\s*(.*)$",
        # 2) 第十二章 标题（中文数字）
        r"^第\s*([零○〇一二三四五六七八九十百千万]+)\s*章[:：]?\s*(.*)$",
        # 3) Chapter 12 标题
        r"^Chapter\s+(\d+)\s*[:：]?\s*(.*)$",
        # 4) Ch 12 标题
        r"^Ch\s+(\d+)\s*[:：]?\s*(.*)$",
        # 5) 第12节 标题
        r"^第\s*([\d]+)\s*节[:：]?\s*(.*)$",
        # 6) 第12话 标题
        r"^第\s*([\d]+)\s*话[:：]?\s*(.*)$",
        # ... 更多模式 ...
    ]

    # 用来存储匹配结果： chapter_num -> title
    directory_map = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 依次尝试每一种pattern
        matched = False
        for pat in patterns:
            match = re.match(pat, line, flags=re.IGNORECASE)
            if match:
                chapter_num_str = match.group(1)
                chapter_title   = match.group(2).strip()

                # 如果是中文数字，需要转换
                # 如果是阿拉伯数字，直接转 int 即可
                if re.match(r"^[零○〇一二三四五六七八九十百千万]+$", chapter_num_str):
                    chapter_num = chinese_to_arabic(chapter_num_str)
                else:
                    chapter_num = int(chapter_num_str)

                directory_map[chapter_num] = chapter_title
                matched = True
                break
        
        # 如果已经匹配到其中一个pattern，就不需要继续匹配剩余pattern
        if matched:
            continue

    # 收集需要的章节范围
    chapters_info = []
    for cnum in range(novel_number - range_size, novel_number + range_size + 1):
        if cnum in directory_map:
            if cnum == novel_number:
                chapters_info.append(f"【当前】第{cnum}章：{directory_map[cnum]}")
            else:
                chapters_info.append(f"第{cnum}章：{directory_map[cnum]}")

    if chapters_info:
        return "\n".join(chapters_info)
    return ""

def generate_chapter_with_state(
    novel_settings: str,
    novel_novel_directory: str,
    api_key: str,
    base_url: str,
    model_name: str,
    novel_number: int,
    filepath: str,
    word_number: int,
    lastchapter: str,
    user_guidance: str = "",
    temperature: float = 0.7
) -> str:
    """
    多步流程:
      1) 更新/创建全局摘要
      2) 更新/生成角色状态文档
      3) 向量检索获取往期上下文
      4) 从Novel_directory.txt中获取当前(和前后几章)的目录信息
      5) 大纲 -> 正文（可结合用户给出的额外指导）
      6) 写入 chapter_{novel_number}.txt, 更新 last_chapter.txt
      7) 更新向量库

    :param novel_settings:        最终的作品设定（字符串）
    :param novel_novel_directory: 小说目录信息
    :param api_key:              OpenAI API Key
    :param base_url:             OpenAI Base URL
    :param model_name:           LLM 模型名称
    :param novel_number:         当前要生成的章节号
    :param filepath:             文件存放的目录
    :param word_number:          单章目标字数
    :param lastchapter:          上一章内容（若为空字符串，表示无上一章）
    :param user_guidance:        用户对当前章节的额外指导或想法
    :param temperature:          生成温度
    :return:                     本章生成的正文内容
    """
    # 确保文件夹存在
    os.makedirs(filepath, exist_ok=True)

    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature
    )

    # 调试输出函数
    def debug_log(prompt: str, response_content: str):
        """在控制台打印或记录下每次的 Prompt 与 Response，便于观察生成过程。"""
        logging.info(f"\n[Prompt >>>]\n{prompt}\n")
        logging.info(f"[Response <<<]\n{response_content}\n")

    # --- 文件路径定义 ---
    chapters_dir = os.path.join(filepath, "chapters")
    os.makedirs(chapters_dir, exist_ok=True)

    chapter_file = os.path.join(chapters_dir, f"chapter_{novel_number}.txt")
    lastchapter_file = os.path.join(filepath, "last_chapter.txt")
    character_state_file = os.path.join(filepath, "character_state.txt")
    global_summary_file = os.path.join(filepath, "global_summary.txt")

    old_char_state = read_file(character_state_file)
    old_global_summary = read_file(global_summary_file)

    # 1) 更新全局摘要
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

    if lastchapter.strip():
        new_global_summary = update_global_summary(lastchapter, old_global_summary)
    else:
        new_global_summary = old_global_summary

    # 2) 更新角色状态文档
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

    if lastchapter.strip():
        new_char_state = update_character_state(lastchapter, old_char_state)
    else:
        new_char_state = old_char_state

    # 3) 从向量库检索上下文
    relevant_context = get_relevant_context_from_vector_store(
        api_key, base_url, "回顾剧情", k=2
    )

    # 4) 解析本章及前后章节目录信息
    this_and_related_chapters = parse_chapter_title_from_directory(novel_novel_directory, novel_number, range_size=1)

    # 5) 生成大纲
    def outline_chapter(
        novel_setting: str,
        char_state: str,
        global_summary: str,
        chap_num: int,
        extra_context: str,
        directory_hint: str,
        user_guide: str
    ) -> str:
        """
        将目录提示以及用户额外指导内容一起放入 Prompt 中。
        """
        # 适度修改章节提纲提示词，以整合目录信息 & 用户指导
        outline_prompt = (
            chapter_outline_prompt
            + "\n\n【目录参考】\n" + directory_hint
            + "\n\n【用户指导】\n" + user_guide
        ).format(
            novel_setting=novel_setting,
            character_state=char_state + "\n\n【历史上下文】\n" + extra_context,
            global_summary=global_summary,
            novel_number=chap_num
        )

        response = model.invoke(outline_prompt)
        if not response:
            logging.warning("outline_chapter: No response.")
            return ""
        debug_log(outline_prompt, response.content)
        return response.content.strip()

    chap_outline = outline_chapter(
        novel_settings, new_char_state, new_global_summary, novel_number,
        relevant_context, this_and_related_chapters, user_guidance
    )

    # 6) 生成正文
    def write_chapter(
        novel_setting: str,
        char_state: str,
        global_summary: str,
        outline: str,
        wnum: int,
        extra_context: str,
        directory_hint: str,
        user_guide: str
    ) -> str:
        # 同理，整合目录信息和用户指导
        writing_prompt = (
            chapter_write_prompt
            + "\n\n【目录参考】\n" + directory_hint
            + "\n\n【用户指导】\n" + user_guide
        ).format(
            novel_setting=novel_setting,
            character_state=char_state + "\n\n【历史上下文】\n" + extra_context,
            global_summary=global_summary,
            chapter_outline=outline,
            word_number=wnum
        )

        response = model.invoke(writing_prompt)
        if not response:
            logging.warning("write_chapter: No response.")
            return ""
        debug_log(writing_prompt, response.content)
        return response.content.strip()

    chapter_content = write_chapter(
        novel_settings,
        new_char_state,
        new_global_summary,
        chap_outline,
        word_number,
        relevant_context,
        this_and_related_chapters,
        user_guidance
    )

    # 写入文件并更新记录
    if chapter_content:
        save_string_to_txt(chapter_content, chapter_file)

        # 更新 last_chapter.txt
        clear_file_content(lastchapter_file)
        save_string_to_txt(chapter_content, lastchapter_file)

        # 更新角色状态、全局摘要
        clear_file_content(character_state_file)
        save_string_to_txt(new_char_state, character_state_file)

        clear_file_content(global_summary_file)
        save_string_to_txt(new_global_summary, global_summary_file)

        # 7) 更新向量检索库
        update_vector_store(api_key, base_url, chapter_content)
        logging.info(f"Chapter {novel_number} generated successfully.")
    else:
        logging.warning(f"Chapter {novel_number} generation failed.")

    return chapter_content

def import_knowledge_file(api_key: str, base_url: str, file_path: str) -> None:
    """
    将用户选定的文本文件导入到向量库，以便在写作时检索。
    可以在UI中提供按钮来调用此函数。
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

    :param content: 原始文本内容
    :param similarity_threshold: 相邻句子合并的语义相似度阈值，小于此值则会开启新的段落
    :param max_length: 每个段落的最大长度（按字符数计算，超过则进一步拆分）
    :return: 切分好的段落列表
    """

    # 1. 按句子切分
    nltk.download('punkt', quiet=True)  # 确保 punkt 数据可用
    sentences = nltk.sent_tokenize(content)

    if not sentences:
        return []

    # 2. 加载 SentenceTransformer 模型，用于计算语义相似度
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(sentences)

    # 3. 根据相邻句子的语义相似度合并段落
    merged_paragraphs = []
    current_sentences = [sentences[0]]
    current_embedding = embeddings[0]

    for i in range(1, len(sentences)):
        sim = cosine_similarity([current_embedding], [embeddings[i]])[0][0]
        if sim >= similarity_threshold:
            # 语义相似则并入当前段落
            current_sentences.append(sentences[i])
            # 更新current_embedding为合并后的平均值（可选，也可只采用最后一句做比较）
            current_embedding = (current_embedding + embeddings[i]) / 2.0
        else:
            # 语义相似度不足，另起一个新段落
            merged_paragraphs.append(" ".join(current_sentences))
            current_sentences = [sentences[i]]
            current_embedding = embeddings[i]

    # 把最后一段加进去
    if current_sentences:
        merged_paragraphs.append(" ".join(current_sentences))

    # 4. 根据最大长度 max_length 做二次拆分，避免段落过长
    final_segments = []
    for para in merged_paragraphs:
        # 如果段落长度超过max_length，进一步切分
        if len(para) > max_length:
            sub_segments = split_by_length(para, max_length=max_length)
            final_segments.extend(sub_segments)
        else:
            final_segments.append(para)

    # 返回最终段落列表
    return final_segments


def split_by_length(text: str, max_length: int = 500) -> List[str]:
    """
    将文本按照max_length进行拆分，以避免段落过长。
    这里以字符数为单位进行简单的拆分，也可以改为按词数或token数等。
    """
    segments = []
    start_idx = 0
    while start_idx < len(text):
        end_idx = min(start_idx + max_length, len(text))
        segment = text[start_idx:end_idx]
        segments.append(segment.strip())
        start_idx = end_idx
    return segments
