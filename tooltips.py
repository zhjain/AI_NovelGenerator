# tooltips.py
# -*- coding: utf-8 -*-

tooltips = {
    "api_key": "在这里填写你的API Key。如果使用OpenAI官方接口，请在 https://platform.openai.com/account/api-keys 获取。",
    "base_url": "模型的接口地址。若使用OpenAI官方：https://api.openai.com/v1。若使用Ollama本地部署，则类似 http://localhost:11434/v1。",
    "interface_format": "指定LLM接口兼容格式，可选OpenAI、Ollama、ML Studio等。",
    "model_name": "要使用的模型名称，例如gpt-3.5-turbo、llama2等。如果是Ollama，请填写你下载好的本地模型名。",
    "temperature": "生成文本的随机度。数值越大越具有发散性，越小越严谨。",
    "max_tokens": "限制单次生成的最大Token数。范围1~100000，请根据模型上下文及需求填写合适值。",
    "embedding_api_key": "调用Embedding模型时所需的API Key。",
    "embedding_interface_format": "Embedding模型接口风格，比如OpenAI或Ollama。",
    "embedding_url": "Embedding模型接口地址。",
    "embedding_model_name": "Embedding模型名称，如text-embedding-ada-002。",
    "embedding_retrieval_k": "向量检索时返回的Top-K结果数量。",
    "topic": "小说的大致主题或主要故事背景描述。",
    "genre": "小说的题材类型，如玄幻、都市、科幻等。",
    "num_chapters": "小说期望的章节总数。",
    "word_number": "每章的目标字数。",
    "filepath": "生成文件存储的根目录路径。所有txt文件、向量库等放在该目录下。",
    "chapter_num": "当前正在处理的章节号，用于生成草稿或定稿操作。",
    "user_guidance": "为本章提供的一些额外指令或写作引导。",
    "characters_involved": "本章需要重点描写或影响剧情的角色名单。",
    "key_items": "在本章中出现的重要道具、线索或物品。",
    "scene_location": "本章主要发生的地点或场景描述。",
    "time_constraint": "本章剧情中涉及的时间压力或时限设置。"
}
