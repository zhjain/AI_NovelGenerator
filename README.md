**自动小说生成工具**

**核心功能：**

✅ **小说设定**（世界观、人物、剧情结构）  
✅ **章节生成**（多步生成，确保逻辑连贯）  
✅ **角色状态 & 伏笔管理**（追踪人物发展）  
✅ **向量检索**（保证长篇小说上下文一致）  
✅ **一致性检查**（防止剧情冲突）  
✅ **GUI 友好交互**（可配置 & 直观操作）  

# **部署与使用指南**

## **1. 环境要求**
在开始之前，请确保你的系统满足以下要求：
- **Python 3.8+**
- **pip 已安装**（Python 包管理器）
- **API 访问权限**（支持OpenAI API方式的任何AI，中文推荐使用DeepSeek）

---

## **2. 安装依赖**

**进入项目目录，执行**：
```bash
pip install -r requirements.txt
```
### 安装语句切分模型punkt（可选，默认程序运行后会自动加载）
**Python环境终端输入**:
```bash
python
import nltk
nltk.download('punkt')
```

等待下载完成(很小，下载很快的)

**至此，环境配置完成。**

---

## **3. 项目结构**
克隆或下载本项目后，你会看到如下结构：
```plaintext
.
├── main.py                    # 入口文件，运行 GUI
├── ui.py                      # 图形界面
├── novel_generator.py         # 章节生成核心逻辑
├── consistency_checker.py     # 一致性检查 (防止剧情冲突)
├── prompt_definitions.py      # 预定义的 AI 提示词
├── utils.py                   # 通用工具函数 (文件操作)
├── config_manager.py          # 处理配置信息 (API Key, Base URL)
├── config.json                # 用户配置文件 (可选)
└── vectorstore/               # (可选) 存储向量数据库
```

---

## **4. 配置 API Key**
运行前，你需要**配置 API Key** 以便调用 OpenAI 或本地 LLM。

### **方式 1：手动修改 `config.json`**
在 `config.json` 文件中填入：
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

### **方式 2：通过 GUI 直接输入**
- 启动程序后，在 GUI 中输入 `API Key` 并选择 `Base URL`，然后**点击 "保存配置"** 以存储到 `config.json`。

---

## **5. 运行程序**
### **方式 1：使用 `Python` 运行**
```bash
python main.py
```
程序启动后，你会看到一个图形界面，方便用户交互。

### **方式 2：打包成可执行文件**
如果你希望**打包成可执行文件**（避免 Python 依赖），可以使用 `PyInstaller`：
```bash
pip install pyinstaller
pyinstaller --onefile --windowed main.py
```
这样会在 `dist/` 目录下生成 `main.exe`（Windows）或 `main`（Linux/macOS）。

或者使用提供的`main.spec`，执行以下打包指令：
```bash
pyinstaller main.spec
```

---

## **6. 使用指南**
### **步骤 1：设置小说参数**
在 GUI 界面：
1. **输入 API Key & Base URL**（或使用 `config.json`）。
2. **选择模型**（如 `gpt-4o`）。
3. **输入小说主题**（如 "未来世界中的 AI 革命"）。
4. **选择小说类型**（如 "科幻"、"奇幻"、"悬疑"）。
5. **设置章节数和每章字数**（如 10 章，每章 3000 字）。
6. **选择存储路径**（建议创建 `novels/` 目录）。

### **步骤 2：生成小说设定 & 目录**
点击 **"1. 生成设定 & 目录"**，系统将：
- 生成**世界观设定**（`Novel_setting.txt`）。
- 生成**章节目录**（`Novel_directory.txt`）。

### **步骤 3：生成章节**
点击 **"2. 生成单章"**，系统将：
- 读取**上一章节内容**（`lastchapter.txt`）。
- 通过**向量检索**查找相关背景信息。
- **动态调整角色状态**（`character_state.txt`）。
- **生成完整章节**，并保存到 `chapters/chapter_X.txt`。

### **步骤 4：一致性检查（可选）**
点击 **"3. 一致性审校"**，系统将：
- 检查**角色行为、剧情逻辑**是否前后矛盾。
- 识别是否有**未解伏笔**，保证故事合理性。

---

## **7. 生成文件管理**
所有生成的文件存储在你选择的目录下：
```plaintext
output_directory/
├── Novel_setting.txt         # 小说世界观 & 角色设定
├── Novel_directory.txt       # 章节目录
├── lastchapter.txt           # 最新章节 (供 AI 参考)
├── character_state.txt       # 角色状态 (道具、情感、技能)
├── global_summary.txt        # 小说摘要
└── chapters/                 # 所有章节
    ├── chapter_1.txt
    ├── chapter_2.txt
    ├── chapter_3.txt
    └── ...
```

---

## **8. 可能遇到的问题**
### **1. `Chroma' object has no attribute 'persist'`**
**原因：** `Chroma` 版本问题。  
**解决方案：**
```bash
pip uninstall chromadb
pip install chromadb==0.3.21  # 或尝试其他版本
```
如果仍然报错，可以在 `novel_generator.py` 里**注释 `store.persist()`**。

---

### **2. 生成内容不符合预期**
**可能的原因：**
- 主题不够清晰，可以在 `topic` 字段中添加详细设定（如 `“废土世界 + AI 叛乱”`）。
- 角色设定较少，可手动在 `Novel_setting.txt` 里补充。
