# chapter_directory_parser.py
# -*- coding: utf-8 -*-
import re

def get_chapter_info_from_directory(novel_directory_content: str, chapter_number: int):
    """
    从给定的 novel_directory_content 文本中，解析 “第X章” 行，并提取本章的标题和可能的简述。
    返回一个 dict: {
       "chapter_title": <字符串>,
       "chapter_brief": <字符串> (若没有则为空)
    }
    注意：目录文本示例格式:
        第1章 ：潮起
        第2章 ：阴影浮现 - 主要角色冲突爆发
        ...
    也可能没有简述，只有一个简单标题。
    """

    # 将文本逐行拆分
    lines = novel_directory_content.splitlines()

    # 章节匹配：形如 “第5章 ：xxx” or “第5章: xxx” or “第5章 xxx”
    pattern = re.compile(r'^第\s*(\d+)\s*章\s*[:：]?\s*(.*)$')

    for line in lines:
        match = pattern.match(line.strip())
        if match:
            chap_num = int(match.group(1))
            if chap_num == chapter_number:
                # group(2) 可能是标题及简述的混合
                full_title = match.group(2).strip()
                # 这里假设用 '-' 进一步区分“标题 - 简述”，也可能用户没写“ - ”
                if ' - ' in full_title:
                    # 根据你的目录格式自由处理
                    parts = full_title.split(' - ', 1)
                    return {
                        "chapter_title": parts[0].strip(),
                        "chapter_brief": parts[1].strip()
                    }
                else:
                    return {
                        "chapter_title": full_title,
                        "chapter_brief": ""
                    }

    # 如果没有匹配到，返回默认
    return {
        "chapter_title": f"第{chapter_number}章",
        "chapter_brief": ""
    }
