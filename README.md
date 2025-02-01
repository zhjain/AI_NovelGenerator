
# **自动小说生成工具**

> 一款基于大模型的多功能小说生成器，帮助你快速生成连贯、可控、可审校的长篇故事。

<div align="center">

| 功能点 | 描述 |
|-------|-----|
| 🚀 **小说设定** | 世界观 / 人物 / 剧情结构 |
| 📝 **章节生成** | 多步生成，确保逻辑连贯 |
| 👥 **角色状态 & 伏笔管理** | 追踪人物发展、重要事件 |
| 🔍 **向量检索** | 保证长篇小说上下文一致 |
| 📁 **自定义知识库** | 可上传本地参考文档 |
| ✅ **一致性检查** | 自动识别剧情冲突或矛盾 |
| 🖥 **GUI 友好交互** | 所见即所得，配置 & 操作简便 |

</div>

---

## **目录**
1. [环境要求](#环境要求)  
2. [安装依赖](#安装依赖)  
3. [项目结构](#项目结构)  
4. [配置 API Key](#配置-api-key)  
5. [运行程序](#运行程序)  
6. [使用指南](#使用指南)  
7. [生成文件管理](#生成文件管理)  
8. [常见问题](#常见问题)  

---

## **环境要求**
在开始之前，请确保你的系统满足以下条件：
- 推荐**Python 3.10+**  
- 已安装 **pip**（Python 包管理器）  
- 拥有 **API Key**（如 OpenAI 或 DeepSeek）或 **支持OpenAI调用方式的本地接口**

---

## **安装依赖**
1. 在项目根目录下，打开终端或命令行  
2. 执行以下命令安装所有依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. （可选）若需要手动安装 `nltk` 的 `punkt_tab` 数据包：
   ```bash
   python
   >>> import nltk
   >>> nltk.download('punkt_tab')
   ```

---

## **项目结构**
克隆或下载本项目后，你将看到如下目录结构：

```
.
├── main.py                # 入口文件, 运行 GUI
├── ui.py                  # 图形界面
├── novel_generator.py     # 章节生成核心逻辑
├── consistency_checker.py # 一致性检查, 防止剧情冲突
|—— chapter_directory_parser.py #格式化目录
├── prompt_definitions.py  # 定义 AI 提示词
├── utils.py               # 常用工具函数, 文件操作
├── config_manager.py      # 管理配置 (API Key, Base URL)
├── config.json            # 用户配置文件 (可选)
└── vectorstore/           # (可选) 本地向量数据库存储
```

> `vectorstore/` 文件夹将在程序运行后自动生成，存储向量检索的缓存数据。

---

## **配置 API Key**
你可以通过以下方式指定或修改 API Key：

### 方式 1：**修改 `config.json`**
打开 `config.json`，将对应字段替换为你的配置：
```json
{
    "api_key": "your_openai_api_key",
    "base_url": "https://api.openai.com/v1",
    "model_name": "gpt-4o",
    "topic": "未来科技",
    "genre": "科幻",
    "num_chapters": 10,
    "word_number": 3000,
    "filepath": "output_directory"
}
```

### 方式 2：**通过 GUI 输入**
1. 运行 `main.py` 后会弹出图形界面  
2. 在界面输入 `API Key`、`Base URL`、模型名称、Temperature 等  
3. 点击 **“保存配置”** 按钮，自动生成或更新 `config.json`

---

## **运行程序**

### **方式 1：使用 Python 解释器**
```bash
python main.py
```
执行后，GUI 将会启动，你可以在图形界面中进行各项操作。

### **方式 2：打包为可执行文件**
如果你想在无 Python 环境的机器上使用本工具，可以使用 **PyInstaller** 进行打包：

```bash
pip install pyinstaller
pyinstaller main.spec
```
打包完成后，会在 `dist/` 目录下生成可执行文件（如 Windows 下的 `main.exe`）。

---

## **使用指南**

1. **启动后，先完成基本参数设置：**  
   - **API Key & Base URL**（如 `https://api.openai.com/v1`）  
   - **模型名称**（如 `gpt-3.5-turbo`、`gpt-4o` 等）
   - **Temperature**(0~1，决定文字创意程度)  
   - **主题(Topic)**（如 “废土世界的 AI 叛乱”）
   - **类型(Genre)**（如 “科幻”/“魔幻”/“都市幻想”）
   - **章节数**、**每章字数**（如 10 章，每章约 3000 字）
   - **保存路径**（建议创建一个新的输出文件夹）

2. **点击「1. 生成设定 & 目录」**  
   - 系统将基于主题、类型等生成：
     - `Novel_setting.txt`：**世界观 & 整体设定**  
     - `Novel_directory.txt`：**章节目录**（含标题、简要提示）  
   - 可以在生成后的文件中查看、修改或补充世界观设定和目录标题。

3. **点击「2. 生成章节草稿」**  
   - 在生成章节之前，你可以：
     - **设置章节号**（如要写第 1 章，就填 `1`）  
     - **在“本章指导”输入框**中，填写对本章剧情的任何期望或指导  
   - 点击按钮后，系统将：
     - 自动读取前文与 `Novel_directory.txt` 的标题与简述  
     - 调用向量检索回顾剧情，确保上下文连贯  
     - 生成本章大纲 (`outline_X.txt`) 及正文 (`chapter_X.txt`)  
   - 生成完成后，可在左侧查看本章草稿内容。

4. **手动检查 & 编辑**（可选）  
   - 你可以在文本编辑器中修改 `chapter_X.txt` 的内容，使之更符合个人审美或剧情要求。

5. **点击「3. 定稿当前章节」**  
   - 系统将：
     - **更新全局摘要**（`global_summary.txt`）  
     - **更新角色状态**（`character_state.txt`，包含人物发展、物品变动、剧情线索等）  
     - **更新向量检索库**，保证后续章节能够调用最新信息。

6. **一致性检查（可选）**  
   - 点击「4. 一致性审校」进行冲突检测，比如**角色逻辑、剧情前后矛盾**等。  
   - 若有冲突，会在日志区输出详细说明。

7. **重复第 3~5 步** 直到所有章节完成！

---

## **生成文件管理**

在你指定的输出文件夹中，程序会自动生成并维护以下文件/目录：

```
output_directory/
├── Novel_setting.txt       # 世界观 & 整体设定
├── Novel_directory.txt     # 小说章节目录
├── character_state.txt     # 角色状态/物品/伏笔等追踪
├── global_summary.txt      # 整体剧情摘要 (用于后续上下文)
├── outlines/               # 存放各章大纲 (outline_1.txt 等)
├── chapters/               # 存放每一章成稿 (chapter_1.txt 等)
└── vectorstore/            # 向量检索数据库 (可清空重置)
```

- **`character_state.txt`**：记录角色的动机、能力、持有物品等关键信息  
- **`global_summary.txt`**：每次定稿后都会更新，保证剧情可被后续章节引用  
- **`chapters/`**：每一章的正文会独立保存，方便你随时手动修改

---

## **常见问题**

#### 1. **`Chroma' object has no attribute 'persist'` 错误**

- **原因**：Chroma 库版本不匹配  
- **解决办法**：
  ```bash
  pip uninstall chromadb
  pip install chromadb==0.3.21
  ```
  或者在 `novel_generator.py` 中找到 `store.persist()` 并注释掉，如果你的版本不支持此方法。

---

#### 2. **生成内容与预期不符**

- **可能原因**：
  1. 主题或类型过于宽泛，模型难以把握重点  
  2. 角色和世界观设定不够详细，可在 `Novel_setting.txt` 中增补更多细节  
- **建议**：
  - 在 GUI 的“本章指导”输入框中填写更明确的剧情走向或重点描述，让模型更好地理解和发挥

---

若你还有其他问题或需求，欢迎在项目 Issues 中提出。