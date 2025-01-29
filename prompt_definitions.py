"""
集中存放所有提示词(Prompt)，便于统一管理和修改。
"""

# ===============  提示词：设定 & 目录  ===================

set_prompt = """\
请根据主题:{topic}、类型:{genre}、章数:{number_of_chapters}、每章字数:{word_number}来完善小说整体设定。
需要包含以下信息：
1. 小说名称、总字数走向（大致范围即可）。
2. 小说类型与基调（如：都市、穿越、战争等类型，以及轻松、爆笑、暗黑等基调）。
3. 写作风格（正式 / 轻松；细腻 / 简洁；抒情 / 客观；叙事视角等）。
4. 整体世界观（时间背景、地理环境、社会结构、科技或魔法水平、重要历史传说或事件等）。
5. 核心内容梗概（可以使用常见叙事结构，如三幕结构、英雄之旅等）。
6. 初步的情节安排设想（主线、副线、交织等关键思路）。
7. 初步的人物关系与主要角色设定（角色定位、主要冲突或关系）。
8. 结尾可能的方向（圆满、悲剧、开放式等）。

请按照上述要点详细输出，但不用标数字。要清晰、有逻辑、有条理。
"""

character_prompt = """\
基于已生成的小说整体设定：
{novel_setting}
请你完善以下内容，帮助我们更好地维持人物形象和成长轨迹：
1. 列出核心角色（至少3个），并对每个角色进行详细性格特征描述。
2. 强调每个角色的潜在内心冲突、目标与动机。
3. 为每个角色添加至少一个“暗线”或隐藏秘密，以及在故事进行中如何可能被揭示的思路。
4. 指出主要角色之间的关键关系和冲突点，为后续情节埋下伏笔。
"""

dark_lines_prompt = """\
在当前设定中已出现以下角色与背景：
{character_info}
请帮助我们构思若干暗线、伏笔或隐藏冲突，以便在后续章节中逐渐揭示并影响故事走向。要求：
1. 每个暗线至少说明其最初的表现、发展走向，以及揭示或爆发的条件。
2. 这些暗线可以与角色背景、世界观、关键事件等有关。
3. 需注意保留悬念，与已知设定不冲突。
4. 在后续创作中可多次提及这些暗线，并在中后期通过角色行为或剧情变化逐步揭示。
"""

finalize_setting_prompt = """\
请基于以下信息，整合并输出最终的《小说设定》：
1. 之前的“整体设定”：
{novel_setting_base}
2. 扩充的“角色设定”：
{character_setting}
3. 暗线与伏笔构思：
{dark_lines}

要求：
1. 结构清晰，将以上内容融合为一个完整的设定说明。
2. 着重强调角色与暗线的衔接、世界观与角色动机的结合，方便后续写作保持前后一致。
3. 语言通畅，不使用Markdown格式，直接输出文本内容。
"""

novel_directory_prompt = """\
根据以下最终《小说设定》：
{final_novel_setting}
并按照下面的小说目录模板生成 {number_of_chapters} 章的目录，同时确保目录符合小说设定中的叙事结构、角色发展及暗线伏笔。
目录模板：
第1章 ：< text >
第2章 ：< text >
...
第{number_of_chapters}章 ：< text >

请严格按照上述格式输出每一章的名称，且勿使用Markdown语法。
"""

# ===============  提示词：章节+角色状态流程 ===================

summary_prompt = """\
这是新生成的章节文本:
{chapter_text}

这是当前的全局摘要(可能为空):
{global_summary}

请在不超过1000字的前提下，基于当前全局摘要和本章新增剧情，更新全局摘要。
保留原有重要信息，并融入本章的新内容。
不要透露结局，不要过度展开未来剧情。
"""

update_character_state_prompt = """\
这是新生成的章节文本:
{chapter_text}

这是当前角色状态文档(可能为空):
{old_state}

请更新角色状态，包括：
1. 角色持有的物品或能力变化。
2. 角色间关系、冲突或合作的新动向。
3. 正在发生的重要事件列表，有无进展或新事件产生。
4. 任意新增角色或出场人物等。
5. 请保证结构完整，能在后续章节继续引用。

使用简洁、易读的方式描述，可用条目或段落表示。保持与旧文档风格一致。
"""

chapter_outline_prompt = """\
以下是当前小说设定与角色状态信息：
- 小说设定：{novel_setting}
- 角色状态：{character_state}
- 全局摘要：{global_summary}
- 本章节编号：第 {novel_number} 章

请为即将写作的 第{novel_number}章 设计一个简要大纲：
1. 本章的主要冲突或事件？
2. 哪些角色会出现？情感与目标变化？
3. 如何进一步暗示或推动暗线和角色冲突？
4. 如何结尾留下悬念？

直接用1、2、3、4分点说明即可。
"""

chapter_write_prompt = """\
下面是该章写作所需信息：
1. 小说设定：{novel_setting}
2. 角色状态：{character_state}
3. 全局摘要：{global_summary}
4. 本章大纲：{chapter_outline}

请写出本章节的完整正文：
1. 确保本章字数不少于 {word_number} 字。
2. 不要使用分节标题，直接整体输出正文。
3. 可以着重描写人物心理、环境氛围等，以保证足够长度。
4. 在结尾部分保留一定悬念或剧情转折，为下一章做铺垫。
"""
