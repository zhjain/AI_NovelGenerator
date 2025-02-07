# novel_generator.py
# -*- coding: utf-8 -*-
import os
import logging
import re
import time
import traceback
import json
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
    chunked_chapter_blueprint_prompt,
    summary_prompt,
    update_character_state_prompt,
    first_chapter_draft_prompt,
    next_chapter_draft_prompt,
    summarize_recent_chapters_prompt
)

# 章节目录解析
from chapter_directory_parser import get_chapter_info_from_blueprint

from llm_adapters import create_llm_adapter
from embedding_adapters import create_embedding_adapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ============ 进度文件管理 ============

PROGRESS_FILE = "progress.json"

def load_progress() -> dict:
    """
    简易进度文件读取，如果不存在则返回默认空字典。
    你也可以在这里定制更多的进度信息。
    """
    if not os.path.exists(PROGRESS_FILE):
        return {
            "architecture_done": False,
            "blueprint_done": False,
            "blueprint_chunk_index": 1,  # 若有分块生成，则记录当前分块的起始
            # 也可以记录已完成的章节
            "chapters_generated": [],   # 已经生成草稿的章节列表
            "chapters_finalized": []    # 已经定稿的章节列表
        }
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "architecture_done": False,
            "blueprint_done": False,
            "blueprint_chunk_index": 1,
            "chapters_generated": [],
            "chapters_finalized": []
        }

def save_progress(progress: dict):
    """
    将进度写入到 progress.json 中。
    """
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# ============ 通用的重试封装 ============

def call_with_retry(func, max_retries=3, sleep_time=2, fallback_return=None, **kwargs):
    """
    通用的重试机制封装。
    :param func: 要执行的函数
    :param max_retries: 最大重试次数
    :param sleep_time: 重试前的等待秒数
    :param fallback_return: 如果多次重试仍失败时的返回值
    :param kwargs: 传给func的命名参数
    :return: func的结果，若失败则返回 fallback_return
    """
    for attempt in range(1, max_retries + 1):
        try:
            return func(**kwargs)
        except Exception as e:
            logging.warning(f"[call_with_retry] Attempt {attempt} failed with error: {e}")
            traceback.print_exc()
            if attempt < max_retries:
                time.sleep(sleep_time)
            else:
                logging.error("Max retries reached, returning fallback_return.")
                return fallback_return


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
    """
    对 LLM 的调用增加了重试封装，
    如果多次失败，则返回空字符串以继续流程，而不是中断。
    """
    def _invoke(prompt):
        return llm_adapter.invoke(prompt)

    response = call_with_retry(func=_invoke, max_retries=3, fallback_return="", prompt=prompt)
    if not response:
        logging.warning("No response from model after retry. Return empty.")
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
) -> Optional[Chroma]:
    """
    在 filepath 下创建/加载一个 Chroma 向量库并插入 texts。
    如果Embedding失败，则返回 None，不中断任务。
    """
    from langchain.embeddings.base import Embeddings as LCEmbeddings

    store_dir = get_vectorstore_dir(filepath)
    os.makedirs(store_dir, exist_ok=True)

    documents = [Document(page_content=str(t)) for t in texts]

    # 包一层try，如果embedding在初始化或插入过程中报错，则跳过
    try:
        class LCEmbeddingWrapper(LCEmbeddings):
            def embed_documents(self, doc_texts: List[str]) -> List[List[float]]:
                return call_with_retry(
                    func=embedding_adapter.embed_documents,
                    max_retries=3,
                    fallback_return=[],
                    doc_texts=doc_texts
                )

            def embed_query(self, query_text: str) -> List[float]:
                res = call_with_retry(
                    func=embedding_adapter.embed_query,
                    max_retries=3,
                    fallback_return=[],
                    query_text=query_text
                )
                return res

        chroma_embedding = LCEmbeddingWrapper()

        vectorstore = Chroma.from_documents(
            documents,
            embedding=chroma_embedding,
            persist_directory=store_dir,
            client_settings=Settings(anonymized_telemetry=False),
            collection_name="novel_collection"
        )
        return vectorstore
    except Exception as e:
        logging.warning(f"Init vector store failed: {e}")
        traceback.print_exc()
        return None

def load_vector_store(
    embedding_adapter,
    filepath: str
) -> Optional[Chroma]:
    """
    读取已存在的 Chroma 向量库。若不存在则返回 None。
    如果加载失败（embedding 或IO问题），则返回 None。
    """
    store_dir = get_vectorstore_dir(filepath)
    if not os.path.exists(store_dir):
        logging.info("Vector store not found. Will return None.")
        return None

    from langchain.embeddings.base import Embeddings as LCEmbeddings

    try:
        class LCEmbeddingWrapper(LCEmbeddings):
            def embed_documents(self, doc_texts: List[str]) -> List[List[float]]:
                return call_with_retry(
                    func=embedding_adapter.embed_documents,
                    max_retries=3,
                    fallback_return=[],
                    doc_texts=doc_texts
                )

            def embed_query(self, query_text: str) -> List[float]:
                res = call_with_retry(
                    func=embedding_adapter.embed_query,
                    max_retries=3,
                    fallback_return=[],
                    query_text=query_text
                )
                return res

        chroma_embedding = LCEmbeddingWrapper()

        return Chroma(
            persist_directory=store_dir,
            embedding_function=chroma_embedding,
            client_settings=Settings(anonymized_telemetry=False),
            collection_name="novel_collection"
        )
    except Exception as e:
        logging.warning(f"Failed to load vector store: {e}")
        traceback.print_exc()
        return None


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
    将最新章节文本插入到向量库中。
    若库不存在则初始化；若初始化/更新失败，则跳过。
    """
    splitted_texts = split_text_for_vectorstore(new_chapter)
    if not splitted_texts:
        logging.warning("No valid text to insert into vector store. Skipping.")
        return

    store = load_vector_store(embedding_adapter, filepath)
    if not store:
        logging.info("Vector store does not exist or failed to load. Initializing a new one for new chapter...")
        store = init_vector_store(embedding_adapter, splitted_texts, filepath)
        if not store:
            logging.warning("Init vector store failed, skip embedding.")
        else:
            logging.info("New vector store created successfully.")
        return

    # 如果已有store，则直接往里插入
    try:
        docs = [Document(page_content=str(t)) for t in splitted_texts]
        store.add_documents(docs)
        logging.info("Vector store updated with the new chapter splitted segments.")
    except Exception as e:
        logging.warning(f"Failed to update vector store: {e}")
        traceback.print_exc()


# ============ 向量检索上下文 ============

def get_relevant_context_from_vector_store(
    embedding_adapter,
    query: str,
    filepath: str,
    k: int = 2
) -> str:
    """
    从向量库中检索与 query 最相关的 k 条文本，拼接后返回。
    如果向量库加载/检索失败，则返回空字符串。
    """
    store = load_vector_store(embedding_adapter, filepath)
    if not store:
        logging.info("No vector store found or load failed. Returning empty context.")
        return ""

    try:
        docs = store.similarity_search(query, k=k)
        if not docs:
            logging.info(f"No relevant documents found for query '{query}'. Returning empty context.")
            return ""
        combined = "\n".join([d.page_content for d in docs])
        return combined
    except Exception as e:
        logging.warning(f"Similarity search failed: {e}")
        traceback.print_exc()
        return ""


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
    max_tokens: int,
    chapters_text_list: List[str],
    timeout: int = 600
) -> Tuple[str, str]:
    """
    生成 (short_summary, next_chapter_keywords)
    如果解析失败，则返回 (合并文本, "")
    """
    combined_text = "\n".join(chapters_text_list).strip()
    if not combined_text:
        return ("", "")

    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
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
    interface_format: str,
    api_key: str,
    base_url: str,
    llm_model: str,
    topic: str,
    genre: str,
    number_of_chapters: int,
    word_number: int,
    filepath: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: int = 600
) -> None:
    """
    依次调用:
      1. core_seed_prompt
      2. character_dynamics_prompt
      3. world_building_prompt
      4. plot_architecture_prompt
    最终输出 Novel_architecture.txt
    如果已生成，则不重复执行（利用 progress.json 中的标记）。
    """
    progress = load_progress()
    if progress.get("architecture_done", False):
        logging.info("Novel architecture generation is already done. Skip.")
        return

    os.makedirs(filepath, exist_ok=True)

    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=llm_model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
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

    final_content = (
        "#=== 0) 小说设定 ===\n"
        f"主题：{topic},类型：{genre},篇幅：约{number_of_chapters}章（每章{word_number}字）\n\n"
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

    # 更新进度
    progress["architecture_done"] = True
    save_progress(progress)


# ============ 计算分块大小的工具函数 ============

def compute_chunk_size(number_of_chapters: int, max_tokens: int) -> int:
    """
    基于“每章约100 tokens”的粗略估算，
    再结合当前max_tokens，计算分块大小：
      chunk_size = (floor(max_tokens/100/10)*10) - 10
    并确保 chunk_size 不会小于1或大于实际章节数。
    """
    tokens_per_chapter = 100.0
    ratio = max_tokens / tokens_per_chapter  # 例如：8192 / 100 = 81.92
    # 先取到最接近的10倍
    ratio_rounded_to_10 = int(ratio // 10) * 10  # => 80
    # 再减10
    chunk_size = ratio_rounded_to_10 - 10       # => 70
    if chunk_size < 1:
        chunk_size = 1
    if chunk_size > number_of_chapters:
        chunk_size = number_of_chapters
    return chunk_size


# ============ 2) 生成章节蓝图（新增分块逻辑） ============

def Chapter_blueprint_generate(
    interface_format: str,
    api_key: str,
    base_url: str,
    llm_model: str,
    filepath: str,
    number_of_chapters: int,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    timeout: int = 600
) -> None:
    """
    如果章节数小于等于 chunk_size，则直接使用 chapter_blueprint_prompt 一次性生成。
    如果章节数较多，则进行分块生成：
      1) 首先说明要生成的总章节数
      2) 先生成 [1..chunk_size] 的章节
      3) 将生成的文本作为已有目录传入，继续生成 [chunk_size+1..] 的章节
      4) 最后汇总全部章节目录写入 Novel_directory.txt

    过程中若发生错误，会进行一定次数重试；若仍失败则保留已生成的结果，方便下次中断续作。
    """
    progress = load_progress()
    if progress.get("blueprint_done", False):
        logging.info("Chapter blueprint generation is already done. Skip.")
        return

    arch_file = os.path.join(filepath, "Novel_architecture.txt")
    if not os.path.exists(arch_file):
        logging.warning("Novel_architecture.txt not found. Please generate architecture first.")
        return

    architecture_text = read_file(arch_file).strip()
    if not architecture_text:
        logging.warning("Novel_architecture.txt is empty.")
        return

    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=llm_model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )

    # 计算分块大小
    chunk_size = compute_chunk_size(number_of_chapters, max_tokens)
    logging.info(f"Number of chapters = {number_of_chapters}, computed chunk_size = {chunk_size}.")

    # 如果一次就可以生成全部
    if chunk_size >= number_of_chapters:
        prompt = chapter_blueprint_prompt.format(
            novel_architecture=architecture_text,
            number_of_chapters=number_of_chapters
        )
        blueprint_text = invoke_with_cleaning(llm_adapter, prompt)
        if not blueprint_text.strip():
            logging.warning("Chapter blueprint generation result is empty.")
            return

        filename_dir = os.path.join(filepath, "Novel_directory.txt")
        clear_file_content(filename_dir)
        save_string_to_txt(blueprint_text, filename_dir)
        logging.info("Novel_directory.txt (chapter blueprint) has been generated successfully (single-shot).")

        progress["blueprint_done"] = True
        save_progress(progress)
        return

    # 否则，分块生成
    final_blueprint = ""
    current_start = progress.get("blueprint_chunk_index", 1)  # 若之前中断，则从上一次的 chunk index 开始
    while current_start <= number_of_chapters:
        current_end = min(current_start + chunk_size - 1, number_of_chapters)

        # 分块提示
        chunk_prompt = chunked_chapter_blueprint_prompt.format(
            novel_architecture=architecture_text,
            chapter_list=final_blueprint,      # 已有的章节列表文本
            number_of_chapters=number_of_chapters,
            n=current_start,
            m=current_end
        )
        logging.info(f"Generating chapters [{current_start}..{current_end}] in a chunk...")

        chunk_result = invoke_with_cleaning(llm_adapter, chunk_prompt)
        if not chunk_result.strip():
            logging.warning(f"Chunk generation for chapters [{current_start}..{current_end}] is empty.")
            chunk_result = ""

        # 将本次生成的文本拼接到最终结果中
        if final_blueprint.strip():
            final_blueprint += "\n\n" + chunk_result
        else:
            final_blueprint = chunk_result

        # 更新下一个块
        current_start = current_end + 1

        # 将当前的 final_blueprint 写入文件，以便中断后保留
        filename_dir = os.path.join(filepath, "Novel_directory.txt")
        clear_file_content(filename_dir)
        save_string_to_txt(final_blueprint.strip(), filename_dir)

        # 更新进度，以便中断后能接着来
        progress["blueprint_chunk_index"] = current_start
        save_progress(progress)

    if not final_blueprint.strip():
        logging.warning("All chunked generation results are empty, cannot create blueprint.")
        return

    # 生成完成
    logging.info("Novel_directory.txt (chapter blueprint) has been generated successfully (chunked).")
    progress["blueprint_done"] = True
    save_progress(progress)


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
    embedding_retrieval_k: int = 2,
    interface_format: str = "openai",
    max_tokens: int = 2048,
    timeout: int = 600
) -> str:
    """
    根据 novel_number 判断是否为第一章。
    - 若是第一章，则使用 first_chapter_draft_prompt
    - 否则使用 next_chapter_draft_prompt
    生成草稿后存入 chapters/chapter_{novel_number}.txt
    """
    progress = load_progress()
    if novel_number in progress.get("chapters_generated", []):
        logging.info(f"Chapter {novel_number} draft already generated. Skip.")
        # 直接返回已有内容
        chapters_dir = os.path.join(filepath, "chapters")
        chapter_file = os.path.join(chapters_dir, f"chapter_{novel_number}.txt")
        return read_file(chapter_file)

    arch_file = os.path.join(filepath, "Novel_architecture.txt")
    novel_architecture_text = read_file(arch_file)

    directory_file = os.path.join(filepath, "Novel_directory.txt")
    blueprint_text = read_file(directory_file)

    global_summary_file = os.path.join(filepath, "global_summary.txt")
    global_summary_text = read_file(global_summary_file)

    character_state_file = os.path.join(filepath, "character_state.txt")
    character_state_text = read_file(character_state_file)

    # 获取本章在目录中的信息
    chapter_info = get_chapter_info_from_blueprint(blueprint_text, novel_number)
    chapter_title = chapter_info["chapter_title"]
    chapter_role = chapter_info["chapter_role"]
    chapter_purpose = chapter_info["chapter_purpose"]
    suspense_level = chapter_info["suspense_level"]
    foreshadowing = chapter_info["foreshadowing"]
    plot_twist_level = chapter_info["plot_twist_level"]
    chapter_summary = chapter_info["chapter_summary"]

    # 准备章节目录文件夹
    chapters_dir = os.path.join(filepath, "chapters")
    os.makedirs(chapters_dir, exist_ok=True)

    # 根据是否是第一章，选择不同的 Prompt
    if novel_number == 1:
        # 使用第一章提示词
        prompt_text = first_chapter_draft_prompt.format(
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

            novel_setting=novel_architecture_text
        )
    else:
        # 若不是第一章，则先获取最近几章文本，并做摘要与检索
        recent_3_texts = get_last_n_chapters_text(chapters_dir, novel_number, n=3)
        short_summary, next_chapter_keywords = summarize_recent_chapters(
            interface_format=interface_format,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            chapters_text_list=recent_3_texts,
            timeout=timeout
        )

        # 从最近章节中获取最后一段内容作为前章结尾
        previous_chapter_excerpt = ""
        for text_block in reversed(recent_3_texts):
            if text_block.strip():
                if len(text_block) > 1500:
                    previous_chapter_excerpt = text_block[-1500:]
                else:
                    previous_chapter_excerpt = text_block
                break

        # 从向量库检索上下文（若失败则为空，不中断）
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

        # 使用后续章节提示词
        prompt_text = next_chapter_draft_prompt.format(
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
            context_excerpt=relevant_context,
            previous_chapter_excerpt=previous_chapter_excerpt
        )

    # 调用LLM生成
    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )
    chapter_content = invoke_with_cleaning(llm_adapter, prompt_text)
    if not chapter_content.strip():
        logging.warning("Generated chapter draft is empty.")

    # 保存章节文本
    chapter_file = os.path.join(chapters_dir, f"chapter_{novel_number}.txt")
    clear_file_content(chapter_file)
    save_string_to_txt(chapter_content, chapter_file)

    logging.info(f"[Draft] Chapter {novel_number} generated as a draft.")

    # 更新进度
    progress["chapters_generated"].append(novel_number)
    save_progress(progress)

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
    embedding_model_name: str,
    interface_format: str,
    max_tokens: int,
    timeout: int = 600
):
    progress = load_progress()
    if novel_number in progress.get("chapters_finalized", []):
        logging.info(f"Chapter {novel_number} is already finalized. Skip.")
        return

    chapters_dir = os.path.join(filepath, "chapters")
    chapter_file = os.path.join(chapters_dir, f"chapter_{novel_number}.txt")
    chapter_text = read_file(chapter_file).strip()
    if not chapter_text:
        logging.warning(f"Chapter {novel_number} is empty, cannot finalize.")
        return

    # 如果内容过短，则尝试扩写
    if len(chapter_text) < 0.7 * word_number:
        chapter_text = enrich_chapter_text(chapter_text, word_number, api_key, base_url, model_name, temperature, interface_format, max_tokens, timeout)
        clear_file_content(chapter_file)
        save_string_to_txt(chapter_text, chapter_file)

    global_summary_file = os.path.join(filepath, "global_summary.txt")
    old_global_summary = read_file(global_summary_file)
    character_state_file = os.path.join(filepath, "character_state.txt")
    old_character_state = read_file(character_state_file)

    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )
    prompt_summary = summary_prompt.format(
        chapter_text=chapter_text,
        global_summary=old_global_summary
    )
    new_global_summary = invoke_with_cleaning(llm_adapter, prompt_summary)
    if not new_global_summary.strip():
        new_global_summary = old_global_summary

    prompt_char_state = update_character_state_prompt.format(
        chapter_text=chapter_text,
        old_state=old_character_state
    )
    new_char_state = invoke_with_cleaning(llm_adapter, prompt_char_state)
    if not new_char_state.strip():
        new_char_state = old_character_state

    clear_file_content(global_summary_file)
    save_string_to_txt(new_global_summary, global_summary_file)

    clear_file_content(character_state_file)
    save_string_to_txt(new_char_state, character_state_file)

    # 更新向量库（若失败则跳过）
    embedding_adapter = create_embedding_adapter(
        embedding_interface_format,
        embedding_api_key,
        embedding_url,
        embedding_model_name
    )
    update_vector_store(embedding_adapter, chapter_text, filepath)

    logging.info(f"Chapter {novel_number} has been finalized.")

    # 更新进度
    progress["chapters_finalized"].append(novel_number)
    save_progress(progress)


def enrich_chapter_text(
    chapter_text: str,
    word_number: int,
    api_key: str,
    base_url: str,
    model_name: str,
    temperature: float,
    interface_format: str,
    max_tokens: int,
    timeout: int=600
) -> str:
    llm_adapter = create_llm_adapter(
        interface_format=interface_format,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
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
        logging.info("Vector store does not exist or load failed. Initializing a new one for knowledge import...")
        store = init_vector_store(embedding_adapter, paragraphs, filepath)
        if store:
            logging.info("知识库文件已成功导入至向量库(新初始化)。")
        else:
            logging.warning("知识库导入失败，跳过。")
    else:
        try:
            docs = [Document(page_content=str(p)) for p in paragraphs]
            store.add_documents(docs)
            logging.info("知识库文件已成功导入至向量库(追加模式)。")
        except Exception as e:
            logging.warning(f"知识库导入失败: {e}")
            traceback.print_exc()
