"""
演示多Agent思路中的“审校Agent”，对最新章节进行简单的一致性或逻辑冲突检查。
可根据需要进行扩展。
"""
from langchain_openai import ChatOpenAI

CONSISTENCY_PROMPT = """\
请检查下面的小说设定与最新章节是否存在明显冲突或不一致之处，如有请列出：
- 小说设定：
{novel_setting}

- 角色状态（可能包含重要信息）：
{character_state}

- 全局摘要：
{global_summary}

- 最新章节内容：
{chapter_text}

如果存在冲突或不一致，请说明；否则请返回“无明显冲突”。
"""

def check_consistency(
    novel_setting: str,
    character_state: str,
    global_summary: str,
    chapter_text: str,
    api_key: str,
    base_url: str,
    model_name: str
) -> str:
    """
    调用模型做简单的一致性检查。可扩展更多提示或校验规则。
    """
    prompt = CONSISTENCY_PROMPT.format(
        novel_setting=novel_setting,
        character_state=character_state,
        global_summary=global_summary,
        chapter_text=chapter_text
    )
    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.3
    )
    response = model.invoke(prompt)
    if not response:
        return "审校Agent无回复"
    return response.content.strip()
