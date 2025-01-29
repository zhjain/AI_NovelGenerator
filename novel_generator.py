import os
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import Dict

from utils import (
    read_file, append_text_to_file, clear_file_content, save_string_to_txt
)
from prompt_definitions import (
    set_prompt, character_prompt, dark_lines_prompt,
    finalize_setting_prompt, novel_directory_prompt,
    summary_prompt, update_character_state_prompt,
    chapter_outline_prompt, chapter_write_prompt
)

# 向量检索相关 (以Chroma为例)，需要安装 langchain, chromadb 等
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

# 默认用此目录存放向量库
VECTOR_STORE_DIR = "vectorstore"

# ===============  多步生成：设置 & 目录  ===============
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
):
    """
    使用多步流程，生成 Novel_setting.txt 与 Novel_directory.txt
    """
    model = ChatOpenAI(
        model=llm_model,
        api_key=api_key,
        base_url=base_url
    )

    def generate_base_setting(state: OverallState):
        prompt = set_prompt.format(
            topic=state["topic"],
            genre=state["genre"],
            number_of_chapters=state["number_of_chapters"],
            word_number=state["word_number"],
        )
        response = model.invoke(prompt)
        if not response:
            return {"novel_setting_base": ""}
        return {"novel_setting_base": response.content.strip()}

    def generate_character_setting(state: OverallState):
        prompt = character_prompt.format(novel_setting=state["novel_setting_base"])
        response = model.invoke(prompt)
        if not response:
            return {"character_setting": ""}
        return {"character_setting": response.content.strip()}

    def generate_dark_lines(state: OverallState):
        prompt = dark_lines_prompt.format(character_info=state["character_setting"])
        response = model.invoke(prompt)
        if not response:
            return {"dark_lines": ""}
        return {"dark_lines": response.content.strip()}

    def finalize_novel_setting(state: OverallState):
        prompt = finalize_setting_prompt.format(
            novel_setting_base=state["novel_setting_base"],
            character_setting=state["character_setting"],
            dark_lines=state["dark_lines"]
        )
        response = model.invoke(prompt)
        if not response:
            return {"final_novel_setting": ""}
        return {"final_novel_setting": response.content.strip()}

    def generate_novel_directory(state: OverallState):
        prompt = novel_directory_prompt.format(
            final_novel_setting=state["final_novel_setting"],
            number_of_chapters=state["number_of_chapters"]
        )
        response = model.invoke(prompt)
        if not response:
            return {"novel_directory": ""}
        return {"novel_directory": response.content.strip()}

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
        print("⚠️ invoke() 结果为空，生成失败。")
        return

    final_novel_setting = result.get("final_novel_setting", "")
    final_novel_directory = result.get("novel_directory", "")
    if not final_novel_setting or not final_novel_directory:
        print("⚠️ 生成失败：缺少 final_novel_setting 或 novel_directory。")
        return

    # 写入文件
    filename_set = os.path.join(filepath, "Novel_setting.txt")
    filename_novel_directory = os.path.join(filepath, "Novel_directory.txt")

    final_novel_setting_cleaned = final_novel_setting.replace('#', '').replace('*', '')
    final_novel_directory_cleaned = final_novel_directory.replace('#', '').replace('*', '')

    append_text_to_file(final_novel_setting_cleaned, filename_set)
    append_text_to_file(final_novel_directory_cleaned, filename_novel_directory)


# ===============  生成章节（含角色状态 & 全局摘要 & 向量检索）  ===============

def init_vector_store(api_key: str, texts: list[str]) -> Chroma:
    """
    初始化并返回一个Chroma向量库，将传入的文本进行嵌入。
    若需要可对 texts 做分句或分块处理；这里只演示简单用法。
    """
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    documents = [Document(page_content=t) for t in texts]
    vectorstore = Chroma.from_documents(documents, embedding=embeddings, persist_directory=VECTOR_STORE_DIR)
    vectorstore.persist()
    return vectorstore

def load_vector_store(api_key: str) -> Chroma:
    """
    读取已存在的向量库。若不存在则返回None或新建一个空的。
    """
    if not os.path.exists(VECTOR_STORE_DIR):
        return None
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings)


def update_vector_store(api_key: str, new_chapter: str):
    """
    将最新章节文本插入到向量库里，用于后续检索参考。
    可根据实际需求做分块处理。此处仅作简单示范。
    """
    store = load_vector_store(api_key)
    if not store:
        # 如果vector store不存在，先初始化
        store = init_vector_store(api_key, [new_chapter])
        return

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    new_doc = Document(page_content=new_chapter)
    store.add_documents([new_doc])
    store.persist()


def get_relevant_context_from_vector_store(api_key: str, query: str, k: int=2) -> str:
    """
    从向量库中检索与 query 最相关的 k 条文本，拼接后返回。
    用于在生成大纲或写正文时，为大模型提供更多上下文。
    """
    store = load_vector_store(api_key)
    if not store:
        return ""
    docs = store.similarity_search(query, k=k)
    # 简单拼接
    combined = "\n".join([d.page_content for d in docs])
    return combined

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
      3) 根据向量检索获取往期章节相关内容
      4) 大纲 -> 正文
      5) 更新向量库
    最终写入 chapter.txt、lastchapter.txt、character_state.txt、global_summary.txt
    """
    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.9
    )

    # --- 文件名定义 ---
    character_state_file = os.path.join(filepath, "character_state.txt")
    global_summary_file = os.path.join(filepath, "global_summary.txt")
    chapter_file = os.path.join(filepath, "chapter.txt")
    lastchapter_file = os.path.join(filepath, "lastchapter.txt")

    # --- 读取现有文档（可能为空） ---
    old_char_state = read_file(character_state_file)
    old_global_summary = read_file(global_summary_file)

    # --- 1) 更新全局摘要 (若上一章文本不为空) ---
    def update_global_summary(chapter_text: str, old_summary: str) -> str:
        prompt = summary_prompt.format(chapter_text=chapter_text, global_summary=old_summary)
        response = model.invoke(prompt)
        if not response:
            return old_summary
        return response.content.strip()

    if lastchapter.strip():
        # 用上一章内容更新全局摘要
        new_global_summary = update_global_summary(lastchapter, old_global_summary)
    else:
        new_global_summary = old_global_summary

    # --- 2) 更新角色状态文档 ---
    def update_character_state(chapter_text: str, old_state: str) -> str:
        prompt = update_character_state_prompt.format(chapter_text=chapter_text, old_state=old_state)
        response = model.invoke(prompt)
        if not response:
            return old_state
        return response.content.strip()

    if lastchapter.strip():
        new_char_state = update_character_state(lastchapter, old_char_state)
    else:
        new_char_state = old_char_state

    # --- 3) 从向量库检索相关上下文，用来帮助生成新的大纲 ---
    # 例如，可以根据“角色状态”或“本章关键词”来查询。
    # 简单示范：以 "回顾剧情" 作为检索Query
    relevant_context = get_relevant_context_from_vector_store(api_key, "回顾剧情", k=2)

    # --- 4) 大纲 -> 正文 ---
    def outline_chapter(novel_setting: str, char_state: str, global_summary: str, chap_num: int, extra_context: str) -> str:
        prompt = chapter_outline_prompt.format(
            novel_setting=novel_setting,
            character_state=char_state + "\n\n【历史上下文】\n" + extra_context,
            global_summary=global_summary,
            novel_number=chap_num
        )
        response = model.invoke(prompt)
        if not response:
            return ""
        return response.content.strip()

    chap_outline = outline_chapter(novel_settings, new_char_state, new_global_summary, novel_number, relevant_context)

    def write_chapter(novel_setting: str, char_state: str, global_summary: str, outline: str, wnum: int, extra_context: str) -> str:
        prompt = chapter_write_prompt.format(
            novel_setting=novel_setting,
            character_state=char_state + "\n\n【历史上下文】\n" + extra_context,
            global_summary=global_summary,
            chapter_outline=outline,
            word_number=wnum
        )
        response = model.invoke(prompt)
        if not response:
            return ""
        return response.content.strip()

    chapter_content = write_chapter(novel_settings, new_char_state, new_global_summary, chap_outline, word_number, relevant_context)

    if chapter_content:
        # --- 写入 chapter.txt 与 lastchapter.txt ---
        append_text_to_file(chapter_content, chapter_file)

        clear_file_content(lastchapter_file)
        save_string_to_txt(chapter_content, lastchapter_file)

        # --- 更新全局摘要、角色状态到文件 ---
        clear_file_content(character_state_file)
        save_string_to_txt(new_char_state, character_state_file)

        clear_file_content(global_summary_file)
        save_string_to_txt(new_global_summary, global_summary_file)

        # --- 5) 更新向量检索库 ---
        update_vector_store(api_key, chapter_content)

    return chapter_content
