import os
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading

from config_manager import load_config, save_config
from utils import read_file
from novel_generator import (
    Novel_novel_directory_generate,
    generate_chapter_with_state
)
from consistency_checker import check_consistency

class NovelGeneratorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Novel Generator GUI - Advanced")

        # 配置持久化
        self.config_file = "config.json"
        self.loaded_config = load_config(self.config_file)

        # 使用 PanedWindow 实现左右布局
        main_pane = ttk.PanedWindow(master, orient="horizontal")
        main_pane.pack(fill="both", expand=True)

        # 左侧：显示区(上下分区)
        self.left_frame = ttk.Frame(main_pane)
        main_pane.add(self.left_frame, weight=3)

        # 右侧：参数输入区
        self.right_frame = ttk.Frame(main_pane, padding="10 10 10 10")
        main_pane.add(self.right_frame, weight=1)

        # 左侧布局：日志区 + 章节内容
        self.build_left_layout()
        # 右侧布局：参数输入区
        self.build_right_layout()

    def build_left_layout(self):
        self.left_frame.rowconfigure(0, weight=1)
        self.left_frame.rowconfigure(1, weight=1)
        self.left_frame.columnconfigure(0, weight=1)

        # 日志区
        log_frame = ttk.LabelFrame(self.left_frame, text="输出日志")
        log_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, width=80, height=10)
        self.log_text.grid(row=0, column=0, sticky="nsew")

        # 章节内容区
        chapter_frame = ttk.LabelFrame(self.left_frame, text="本章内容")
        chapter_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        chapter_frame.rowconfigure(0, weight=1)
        chapter_frame.columnconfigure(0, weight=1)

        self.chapter_result = scrolledtext.ScrolledText(chapter_frame, width=80, height=10, foreground="blue")
        self.chapter_result.grid(row=0, column=0, sticky="nsew")

    def build_right_layout(self, ):
        # 行列配置
        for i in range(12):
            self.right_frame.rowconfigure(i, weight=0)
        self.right_frame.columnconfigure(1, weight=1)

        # 1. API Key
        ttk.Label(self.right_frame, text="API Key:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.api_key_var = tk.StringVar(value=self.loaded_config.get("api_key", ""))
        ttk.Entry(self.right_frame, textvariable=self.api_key_var, width=32).grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # 2. Base URL
        ttk.Label(self.right_frame, text="Base URL:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.base_url_var = tk.StringVar(value=self.loaded_config.get("base_url", "https://api.agicto.cn/v1"))
        ttk.Entry(self.right_frame, textvariable=self.base_url_var, width=32).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # 3. Model Name
        ttk.Label(self.right_frame, text="Model Name:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.model_name_var = tk.StringVar(value=self.loaded_config.get("model_name", "gpt-4o-mini"))
        ttk.Entry(self.right_frame, textvariable=self.model_name_var, width=32).grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # 4. 主题(Topic) 多行输入
        ttk.Label(self.right_frame, text="主题(Topic):").grid(row=3, column=0, padx=5, pady=5, sticky="ne")
        self.topic_text = scrolledtext.ScrolledText(self.right_frame, width=32, height=4)
        self.topic_text.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        topic_default = self.loaded_config.get("topic", "")
        if topic_default:
            self.topic_text.insert(tk.END, topic_default)

        # 5. 类型(Genre)
        ttk.Label(self.right_frame, text="类型(Genre):").grid(row=4, column=0, padx=5, pady=5, sticky="e")
        self.genre_var = tk.StringVar(value=self.loaded_config.get("genre", "玄幻"))
        ttk.Entry(self.right_frame, textvariable=self.genre_var, width=32).grid(row=4, column=1, padx=5, pady=5, sticky="w")

        # 6. 章节数
        ttk.Label(self.right_frame, text="章节数:").grid(row=5, column=0, padx=5, pady=5, sticky="e")
        self.num_chapters_var = tk.IntVar(value=self.loaded_config.get("num_chapters", 10))
        ttk.Entry(self.right_frame, textvariable=self.num_chapters_var, width=8).grid(row=5, column=1, padx=5, pady=5, sticky="w")

        # 7. 每章字数
        ttk.Label(self.right_frame, text="每章字数:").grid(row=6, column=0, padx=5, pady=5, sticky="e")
        self.word_number_var = tk.IntVar(value=self.loaded_config.get("word_number", 3000))
        ttk.Entry(self.right_frame, textvariable=self.word_number_var, width=8).grid(row=6, column=1, padx=5, pady=5, sticky="w")

        # 8. 文件保存路径
        ttk.Label(self.right_frame, text="保存路径:").grid(row=7, column=0, padx=5, pady=5, sticky="e")
        self.filepath_var = tk.StringVar(value=self.loaded_config.get("filepath", ""))
        ttk.Entry(self.right_frame, textvariable=self.filepath_var, width=32).grid(row=7, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(self.right_frame, text="浏览...", command=self.browse_folder).grid(row=7, column=2, padx=5, pady=5, sticky="w")

        # 保存/加载配置按钮
        config_frame = ttk.Frame(self.right_frame)
        config_frame.grid(row=8, column=1, sticky="w")
        ttk.Button(config_frame, text="保存配置", command=self.save_config_btn).grid(row=0, column=0, padx=5)
        ttk.Button(config_frame, text="加载配置", command=self.load_config_btn).grid(row=0, column=1, padx=5)

        # 按钮区域
        row_base = 9
        ttk.Label(self.right_frame, text="章节号:").grid(row=row_base, column=0, sticky="e")
        self.chapter_num_var = tk.IntVar(value=1)
        ttk.Entry(self.right_frame, textvariable=self.chapter_num_var, width=6).grid(row=row_base, column=1, padx=5, pady=5, sticky="w")

        self.btn_generate_full = ttk.Button(self.right_frame, text="1. 生成设定 & 目录", command=self.generate_full_novel)
        self.btn_generate_full.grid(row=row_base+1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        self.btn_generate_chapter = ttk.Button(self.right_frame, text="2. 生成单章(含角色状态)", command=self.generate_chapter_text)
        self.btn_generate_chapter.grid(row=row_base+2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # 可选：添加一个“一致性审校”按钮
        self.btn_check_consistency = ttk.Button(self.right_frame, text="3. 一致性审校", command=self.do_consistency_check)
        self.btn_check_consistency.grid(row=row_base+3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")


    # -------------- 配置管理 --------------
    def load_config_btn(self):
        cfg = load_config(self.config_file)
        if cfg:
            self.api_key_var.set(cfg.get("api_key", ""))
            self.base_url_var.set(cfg.get("base_url", ""))
            self.model_name_var.set(cfg.get("model_name", ""))
            self.genre_var.set(cfg.get("genre", ""))
            self.num_chapters_var.set(cfg.get("num_chapters", 10))
            self.word_number_var.set(cfg.get("word_number", 3000))
            self.filepath_var.set(cfg.get("filepath", ""))

            # 多行文本
            self.topic_text.delete("1.0", tk.END)
            self.topic_text.insert(tk.END, cfg.get("topic", ""))

            self.log("已加载配置。")
        else:
            messagebox.showwarning("提示", "未找到或无法读取配置文件。")

    def save_config_btn(self):
        config_data = {
            "api_key": self.api_key_var.get(),
            "base_url": self.base_url_var.get(),
            "model_name": self.model_name_var.get(),
            "topic": self.topic_text.get("1.0", tk.END).strip(),
            "genre": self.genre_var.get(),
            "num_chapters": self.num_chapters_var.get(),
            "word_number": self.word_number_var.get(),
            "filepath": self.filepath_var.get()
        }
        if save_config(config_data, self.config_file):
            messagebox.showinfo("提示", "配置已保存至 config.json")
            self.log("配置已保存。")
        else:
            messagebox.showerror("错误", "保存配置失败。")

    def browse_folder(self):
        selected_dir = filedialog.askdirectory()
        if selected_dir:
            self.filepath_var.set(selected_dir)

    # -------------- 日志输出 --------------
    def log(self, message: str):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    # -------------- 核心功能按钮 --------------
    def disable_button(self, btn):
        btn.config(state=tk.DISABLED)

    def enable_button(self, btn):
        btn.config(state=tk.NORMAL)

    def generate_full_novel(self):
        """生成小说设定 & 目录"""
        def task():
            self.disable_button(self.btn_generate_full)
            try:
                api_key = self.api_key_var.get().strip()
                base_url = self.base_url_var.get().strip()
                model_name = self.model_name_var.get().strip()
                topic = self.topic_text.get("1.0", tk.END).strip()
                genre = self.genre_var.get().strip()
                num_chapters = self.num_chapters_var.get()
                word_number = self.word_number_var.get()
                filepath = self.filepath_var.get().strip()

                if not filepath:
                    messagebox.showwarning("警告", "请先选择保存文件路径")
                    return

                self.log("开始生成小说设定和目录...")
                Novel_novel_directory_generate(
                    api_key=api_key,
                    base_url=base_url,
                    llm_model=model_name,
                    topic=topic,
                    genre=genre,
                    number_of_chapters=num_chapters,
                    word_number=word_number,
                    filepath=filepath
                )
                self.log("✅ 小说设定和目录生成完成。查看 Novel_setting.txt 和 Novel_directory.txt。")
            except Exception as e:
                self.log(f"❌ 生成小说设定 & 目录时出错: {e}")
            finally:
                self.enable_button(self.btn_generate_full)

        thread = threading.Thread(target=task)
        thread.start()

    def generate_chapter_text(self):
        """多步生成章节：维护全局摘要+角色状态文档，向量检索辅助"""
        def task():
            self.disable_button(self.btn_generate_chapter)
            try:
                api_key = self.api_key_var.get().strip()
                base_url = self.base_url_var.get().strip()
                model_name = self.model_name_var.get().strip()
                novel_number = self.chapter_num_var.get()
                filepath = self.filepath_var.get().strip()
                word_number = self.word_number_var.get()

                # 读取设定 & 目录
                novel_settings_file = os.path.join(filepath, "Novel_setting.txt")
                novel_novel_directory_file = os.path.join(filepath, "Novel_directory.txt")
                last_chapter_file = os.path.join(filepath, "last_chapter.txt")

                novel_settings = read_file(novel_settings_file)
                novel_novel_directory = read_file(novel_novel_directory_file)
                lastchapter = read_file(last_chapter_file)

                if not novel_settings.strip():
                    self.log("⚠️ 未找到 Novel_setting.txt，请先生成设定。")
                    return
                if not novel_novel_directory.strip():
                    self.log("⚠️ 未找到 Novel_directory.txt，请先生成目录。")
                    return

                self.log(f"开始生成第{novel_number}章内容(含角色状态文档更新)...")
                chapter_text = generate_chapter_with_state(
                    novel_settings=novel_settings,
                    novel_novel_directory=novel_novel_directory,
                    api_key=api_key,
                    base_url=base_url,
                    model_name=model_name,
                    novel_number=novel_number,
                    filepath=filepath,
                    word_number=word_number,
                    lastchapter=lastchapter
                )

                if chapter_text:
                    self.log(f"✅ 第{novel_number}章内容生成完成。chapter.txt 已更新。")
                    self.chapter_result.delete("1.0", tk.END)
                    self.chapter_result.insert(tk.END, chapter_text)
                    self.chapter_result.see(tk.END)
                else:
                    self.log("⚠️ 本章生成失败或无内容。")

            except Exception as e:
                self.log(f"❌ 生成章节内容时出错: {e}")
            finally:
                self.enable_button(self.btn_generate_chapter)

        thread = threading.Thread(target=task)
        thread.start()

    def do_consistency_check(self):
        """使用审校Agent对最新章节进行简单一致性或冲突检查"""
        def task():
            self.disable_button(self.btn_check_consistency)
            try:
                api_key = self.api_key_var.get().strip()
                base_url = self.base_url_var.get().strip()
                model_name = self.model_name_var.get().strip()
                filepath = self.filepath_var.get().strip()

                # 读取关键文件
                novel_settings_file = os.path.join(filepath, "Novel_setting.txt")
                character_state_file = os.path.join(filepath, "character_state.txt")
                global_summary_file = os.path.join(filepath, "global_summary.txt")
                last_chapter_file = os.path.join(filepath, "last_chapter.txt")

                novel_setting = read_file(novel_settings_file)
                character_state = read_file(character_state_file)
                global_summary = read_file(global_summary_file)
                last_chapter_text = read_file(last_chapter_file)

                if not last_chapter_text.strip():
                    self.log("⚠️ last_chapter.txt 为空，暂无可检查的章节文本。")
                    return

                self.log("开始一致性审校...")
                result = check_consistency(
                    novel_setting=novel_setting,
                    character_state=character_state,
                    global_summary=global_summary,
                    chapter_text=last_chapter_text,
                    api_key=api_key,
                    base_url=base_url,
                    model_name=model_name
                )
                self.log("审校结果：")
                self.log(result)

            except Exception as e:
                self.log(f"❌ 审校时出错: {e}")
            finally:
                self.enable_button(self.btn_check_consistency)

        thread = threading.Thread(target=task)
        thread.start()
