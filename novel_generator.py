import os
import logging
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

# ============ 日志配置（可选） ============
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
        openai_api_base=base_url  # <-- 这里用传进来的 base_url
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
        openai_api_base=base_url  # <-- 使用 base_url
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
    filepath: str
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
    """
    # 确保文件夹存在
    os.makedirs(filepath, exist_ok=True)

    model = ChatOpenAI(
        model=llm_model,
        api_key=api_key,
        base_url=base_url
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
        return {"novel_setting_base": response.content.strip()}

    def generate_character_setting(state: OverallState) -> Dict[str, str]:
        prompt = character_prompt.format(
            novel_setting=state["novel_setting_base"]
        )
        response = model.invoke(prompt)
        if not response:
            logging.warning("generate_character_setting: No response.")
            return {"character_setting": ""}
        return {"character_setting": response.content.strip()}

    def generate_dark_lines(state: OverallState) -> Dict[str, str]:
        prompt = dark_lines_prompt.format(
            character_info=state["character_setting"]
        )
        response = model.invoke(prompt)
        if not response:
            logging.warning("generate_dark_lines: No response.")
            return {"dark_lines": ""}
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
        return {"novel_directory": response.content.strip()}

    # 构建状态图
    graph = StateGraph(OverallState)
    graph.add_node("generate_base_setting", generate_base_setting)
    graph.add_node("generate_character_setting", generate_character_setting)
    graph.add_node("generate_dark_lines", generate_dark_lines)
    graph.add_node("finalize_novel_setting", finalize_novel_setting)
    graph.add_node("generate_novel_directory", generate_novel_directory)

    # 注意修正此处节点名称
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

    # 清理文本（去除多余 # 或 * 等）
    def clean_text(txt: str) -> str:
        return txt.replace('#', '').replace('*', '')

    final_novel_setting_cleaned = clean_text(final_novel_setting)
    final_novel_directory_cleaned = clean_text(final_novel_directory)

    # 以追加方式保存；如果希望覆盖可改为 save_string_to_txt()
    append_text_to_file(final_novel_setting_cleaned, filename_set)
    append_text_to_file(final_novel_directory_cleaned, filename_novel_directory)

    logging.info("Novel settings and directory generated successfully.")


# ============ 生成章节（每章独立文件） ============

def generate_chapter_with_state(
    novel_settings: str,
    novel_novel_directory: str,
    api_key: str,
    base_url: str,
    model_name: str,
    novel_number: int,
    filepath: str,
    word_number: int,
    lastchapter: str
) -> str:
    """
    多步流程:
      1) 更新/创建全局摘要
      2) 更新/生成角色状态文档
      3) 向量检索获取往期上下文
      4) 大纲 -> 正文
      5) 写入 chapter_{novel_number}.txt, 更新 last_chapter.txt
      6) 更新向量库

    :param novel_settings:        最终的作品设定（字符串）
    :param novel_novel_directory: 小说目录信息（此处暂时未使用，可根据需求做扩展）
    :param api_key:              OpenAI API Key
    :param base_url:             OpenAI Base URL
    :param model_name:           LLM 模型名称
    :param novel_number:         当前要生成的章节号
    :param filepath:             文件存放的目录
    :param word_number:          单章目标字数
    :param lastchapter:          上一章内容（若为空字符串，表示无上一章）
    :return:                     本章生成的正文内容
    """
    # 确保文件夹存在
    os.makedirs(filepath, exist_ok=True)

    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.9
    )

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
        return response.content.strip()

    if lastchapter.strip():
        new_char_state = update_character_state(lastchapter, old_char_state)
    else:
        new_char_state = old_char_state

    # 3) 从向量库检索上下文
    relevant_context = get_relevant_context_from_vector_store(
        api_key, base_url, "回顾剧情", k=2  # <-- 多传一个 base_url
    )

    # 4) 生成大纲
    def outline_chapter(
        novel_setting: str,
        char_state: str,
        global_summary: str,
        chap_num: int,
        extra_context: str
    ) -> str:
        prompt = chapter_outline_prompt.format(
            novel_setting=novel_setting,
            character_state=char_state + "\n\n【历史上下文】\n" + extra_context,
            global_summary=global_summary,
            novel_number=chap_num
        )
        response = model.invoke(prompt)
        if not response:
            logging.warning("outline_chapter: No response.")
            return ""
        return response.content.strip()

    chap_outline = outline_chapter(
        novel_settings, new_char_state, new_global_summary, novel_number, relevant_context
    )

    # 5) 生成正文
    def write_chapter(
        novel_setting: str,
        char_state: str,
        global_summary: str,
        outline: str,
        wnum: int,
        extra_context: str
    ) -> str:
        prompt = chapter_write_prompt.format(
            novel_setting=novel_setting,
            character_state=char_state + "\n\n【历史上下文】\n" + extra_context,
            global_summary=global_summary,
            chapter_outline=outline,
            word_number=wnum
        )
        response = model.invoke(prompt)
        if not response:
            logging.warning("write_chapter: No response.")
            return ""
        return response.content.strip()

    chapter_content = write_chapter(
        novel_settings,
        new_char_state,
        new_global_summary,
        chap_outline,
        word_number,
        relevant_context
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

        # 6) 更新向量检索库
        update_vector_store(api_key, base_url, chapter_content)
        logging.info(f"Chapter {novel_number} generated successfully.")
    else:
        logging.warning(f"Chapter {novel_number} generation failed.")

    return chapter_content
