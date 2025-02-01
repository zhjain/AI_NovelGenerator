# ui.py
# -*- coding: utf-8 -*-
import os
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading

from config_manager import load_config, save_config
from utils import read_file
from novel_generator import (
    Novel_novel_directory_generate,
    generate_chapter_draft,
    finalize_chapter,
    import_knowledge_file,
    clear_vector_store,
    get_last_n_chapters_text,
    summarize_recent_chapters
)
from consistency_checker import check_consistency

class NovelGeneratorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Novel Generator GUI")

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

    def build_right_layout(self):
        # 行列配置
        for i in range(20):
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

        # 4. Temperature
        ttk.Label(self.right_frame, text="Temperature:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
        self.temperature_var = tk.DoubleVar(value=self.loaded_config.get("temperature", 0.7))
        self.temp_value_label = ttk.Label(self.right_frame, text=f"{self.temperature_var.get():.2f}")
        self.temp_value_label.grid(row=3, column=2, padx=5, pady=5, sticky="w")

        temp_scale = ttk.Scale(self.right_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.temperature_var)
        temp_scale.grid(row=3, column=1, padx=5, pady=5, sticky="we")
        def update_temp_label(*args):
            self.temp_value_label.config(text=f"{self.temperature_var.get():.2f}")
        self.temperature_var.trace("w", update_temp_label)

        # 5. 主题(Topic) 多行输入
        ttk.Label(self.right_frame, text="主题(Topic):").grid(row=4, column=0, padx=5, pady=5, sticky="ne")
        self.topic_text = scrolledtext.ScrolledText(self.right_frame, width=32, height=4)
        self.topic_text.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        topic_default = self.loaded_config.get("topic", "")
        if topic_default:
            self.topic_text.insert(tk.END, topic_default)

        # 6. 类型(Genre)
        ttk.Label(self.right_frame, text="类型(Genre):").grid(row=5, column=0, padx=5, pady=5, sticky="e")
        self.genre_var = tk.StringVar(value=self.loaded_config.get("genre", "玄幻"))
        ttk.Entry(self.right_frame, textvariable=self.genre_var, width=32).grid(row=5, column=1, padx=5, pady=5, sticky="w")

        # 7. 章节数
        ttk.Label(self.right_frame, text="章节数:").grid(row=6, column=0, padx=5, pady=5, sticky="e")
        self.num_chapters_var = tk.IntVar(value=self.loaded_config.get("num_chapters", 10))
        ttk.Entry(self.right_frame, textvariable=self.num_chapters_var, width=8).grid(row=6, column=1, padx=5, pady=5, sticky="w")

        # 8. 每章字数
        ttk.Label(self.right_frame, text="每章字数:").grid(row=7, column=0, padx=5, pady=5, sticky="e")
        self.word_number_var = tk.IntVar(value=self.loaded_config.get("word_number", 3000))
        ttk.Entry(self.right_frame, textvariable=self.word_number_var, width=8).grid(row=7, column=1, padx=5, pady=5, sticky="w")

        # 9. 文件保存路径
        ttk.Label(self.right_frame, text="保存路径:").grid(row=8, column=0, padx=5, pady=5, sticky="e")
        self.filepath_var = tk.StringVar(value=self.loaded_config.get("filepath", ""))
        ttk.Entry(self.right_frame, textvariable=self.filepath_var, width=32).grid(row=8, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(self.right_frame, text="浏览...", command=self.browse_folder).grid(row=8, column=2, padx=5, pady=5, sticky="w")

        # 保存/加载配置按钮
        config_frame = ttk.Frame(self.right_frame)
        config_frame.grid(row=9, column=1, sticky="w")
        ttk.Button(config_frame, text="保存配置", command=self.save_config_btn).grid(row=0, column=0, padx=5)
        ttk.Button(config_frame, text="加载配置", command=self.load_config_btn).grid(row=0, column=1, padx=5)

        # 10. 章节号
        ttk.Label(self.right_frame, text="章节号:").grid(row=10, column=0, sticky="e")
        self.chapter_num_var = tk.IntVar(value=1)
        ttk.Entry(self.right_frame, textvariable=self.chapter_num_var, width=6).grid(row=10, column=1, padx=5, pady=5, sticky="w")

        # 11. “用户指导” 多行输入
        ttk.Label(self.right_frame, text="本章指导:").grid(row=11, column=0, padx=5, pady=5, sticky="ne")
        self.user_guide_text = scrolledtext.ScrolledText(self.right_frame, width=32, height=4)
        self.user_guide_text.grid(row=11, column=1, padx=5, pady=5, sticky="w")

        row_base = 12
        # ============ 功能按钮 ============

        # (1) 生成设定 & 目录
        self.btn_generate_full = ttk.Button(self.right_frame, text="Step1. 生成设定 & 目录", command=self.generate_full_novel)
        self.btn_generate_full.grid(row=row_base, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # (2) 生成章节草稿
        self.btn_generate_chapter = ttk.Button(self.right_frame, text="Step2. 生成章节草稿", command=self.generate_chapter_draft_ui)
        self.btn_generate_chapter.grid(row=row_base+1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # (3) 定稿当前章节
        self.btn_finalize_chapter = ttk.Button(self.right_frame, text="Step3. 定稿当前章节", command=self.finalize_chapter_ui)
        self.btn_finalize_chapter.grid(row=row_base+2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # (4) 一致性审校
        self.btn_check_consistency = ttk.Button(self.right_frame, text="[可选]一致性审校", command=self.do_consistency_check)
        self.btn_check_consistency.grid(row=row_base+3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # (5) 导入知识库文件
        self.btn_import_knowledge = ttk.Button(self.right_frame, text="[可选]导入知识库", command=self.import_knowledge_handler)
        self.btn_import_knowledge.grid(row=row_base+4, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # (6) 清空向量库
        self.btn_clear_vectorstore = ttk.Button(self.right_frame, text="清空向量库", command=self.clear_vectorstore_handler)
        self.btn_clear_vectorstore.grid(row=row_base+5, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # (7) 查看剧情要点
        ttk.Button(self.right_frame, text="[查看] 剧情要点", command=self.show_plot_arcs_ui).grid(
            row=row_base+6, column=0, columnspan=2, padx=5, pady=5, sticky="ew"
        )

    # -------------- 配置管理 --------------
    def load_config_btn(self):
        cfg = load_config(self.config_file)
        if cfg:
            self.api_key_var.set(cfg.get("api_key", ""))
            self.base_url_var.set(cfg.get("base_url", ""))
            self.model_name_var.set(cfg.get("model_name", ""))
            self.temperature_var.set(cfg.get("temperature", 0.7))
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
            "temperature": self.temperature_var.get(),
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

    # -------------- 功能 --------------
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
                temperature = self.temperature_var.get()

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
                    filepath=filepath,
                    temperature=temperature
                )
                self.log("✅ 小说设定和目录生成完成。查看 Novel_setting.txt 和 Novel_directory.txt。")
            except Exception as e:
                self.log(f"❌ 生成小说设定 & 目录时出错: {e}")
            finally:
                self.enable_button(self.btn_generate_full)

        thread = threading.Thread(target=task)
        thread.start()

    def generate_chapter_draft_ui(self):
        """生成当前章节的草稿"""
        def task():
            self.disable_button(self.btn_generate_chapter)
            try:
                api_key = self.api_key_var.get().strip()
                base_url = self.base_url_var.get().strip()
                model_name = self.model_name_var.get().strip()
                temperature = self.temperature_var.get()
                filepath = self.filepath_var.get().strip()

                novel_settings_file = os.path.join(filepath, "Novel_setting.txt")
                novel_settings = read_file(novel_settings_file)
                if not novel_settings.strip():
                    self.log("⚠️ 未找到 Novel_setting.txt，请先生成设定。")
                    return

                character_state_file = os.path.join(filepath, "character_state.txt")
                character_state = read_file(character_state_file)
                global_summary_file = os.path.join(filepath, "global_summary.txt")
                global_summary = read_file(global_summary_file)
                novel_directory_file = os.path.join(filepath, "Novel_directory.txt")
                novel_directory = read_file(novel_directory_file)

                chap_num = self.chapter_num_var.get()
                word_number = self.word_number_var.get()
                user_guidance = self.user_guide_text.get("1.0", tk.END).strip()

                # 获取最近3章文本，生成短期摘要
                chapters_dir = os.path.join(filepath, "chapters")
                recent_3_texts = get_last_n_chapters_text(chapters_dir, chap_num, n=3)

                # 用当前模型生成一个较为详细的最近剧情摘要
                model_obj = self.get_llm_model(model_name, api_key, base_url, temperature)
                recent_chapters_summary = summarize_recent_chapters(model_obj, recent_3_texts)

                self.log(f"开始生成第{chap_num}章草稿...")
                draft_text = generate_chapter_draft(
                    novel_settings=novel_settings,
                    global_summary=global_summary,
                    character_state=character_state,
                    recent_chapters_summary=recent_chapters_summary,
                    user_guidance=user_guidance,
                    api_key=api_key,
                    base_url=base_url,
                    model_name=model_name,
                    novel_number=chap_num,
                    word_number=word_number,
                    temperature=temperature,
                    novel_novel_directory=novel_directory,
                    filepath=filepath
                )
                if draft_text:
                    self.log(f"✅ 第{chap_num}章草稿生成完成。请在下方查看。")
                    self.chapter_result.delete("1.0", tk.END)
                    self.chapter_result.insert(tk.END, draft_text)
                    self.chapter_result.see(tk.END)
                else:
                    self.log("⚠️ 本章草稿生成失败或无内容。")

            except Exception as e:
                self.log(f"❌ 生成章节草稿时出错: {e}")
            finally:
                self.enable_button(self.btn_generate_chapter)

        thread = threading.Thread(target=task)
        thread.start()

    def finalize_chapter_ui(self):
        """定稿当前章节：更新全局摘要、角色状态、向量库等"""
        def task():
            self.disable_button(self.btn_finalize_chapter)
            try:
                api_key = self.api_key_var.get().strip()
                base_url = self.base_url_var.get().strip()
                model_name = self.model_name_var.get().strip()
                temperature = self.temperature_var.get()
                filepath = self.filepath_var.get().strip()

                chap_num = self.chapter_num_var.get()
                word_number = self.word_number_var.get()

                self.log(f"开始定稿第{chap_num}章...")
                finalize_chapter(
                    novel_number=chap_num,
                    word_number=word_number,
                    api_key=api_key,
                    base_url=base_url,
                    model_name=model_name,
                    temperature=temperature,
                    filepath=filepath
                )
                self.log(f"✅ 第{chap_num}章定稿完成（已更新全局摘要、角色状态、剧情要点、向量库）。")

                # 读取定稿后的文本显示
                chap_file = os.path.join(filepath, "chapters", f"chapter_{chap_num}.txt")
                final_text = read_file(chap_file)
                self.chapter_result.delete("1.0", tk.END)
                self.chapter_result.insert(tk.END, final_text)
                self.chapter_result.see(tk.END)

            except Exception as e:
                self.log(f"❌ 定稿章节时出错: {e}")
            finally:
                self.enable_button(self.btn_finalize_chapter)

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
                temperature = self.temperature_var.get()
                filepath = self.filepath_var.get().strip()

                # 读取关键文件
                novel_settings_file = os.path.join(filepath, "Novel_setting.txt")
                character_state_file = os.path.join(filepath, "character_state.txt")
                global_summary_file = os.path.join(filepath, "global_summary.txt")
                plot_arcs_file = os.path.join(filepath, "plot_arcs.txt")  # 新增

                novel_setting = read_file(novel_settings_file)
                character_state = read_file(character_state_file)
                global_summary = read_file(global_summary_file)
                plot_arcs = read_file(plot_arcs_file)  # 新增

                # 获取当前章节文本
                chap_num = self.chapter_num_var.get()
                chap_file = os.path.join(filepath, "chapters", f"chapter_{chap_num}.txt")
                chapter_text = read_file(chap_file)

                if not chapter_text.strip():
                    self.log("⚠️ 当前章节文件为空或不存在，无法审校。")
                    return

                self.log("开始一致性审校...")
                result = check_consistency(
                    novel_setting=novel_setting,
                    character_state=character_state,
                    global_summary=global_summary,
                    chapter_text=chapter_text,
                    api_key=api_key,
                    base_url=base_url,
                    model_name=model_name,
                    temperature=temperature,
                    plot_arcs=plot_arcs  # 新增传入
                )
                self.log("审校结果：")
                self.log(result)

            except Exception as e:
                self.log(f"❌ 审校时出错: {e}")
            finally:
                self.enable_button(self.btn_check_consistency)

        thread = threading.Thread(target=task)
        thread.start()

    def import_knowledge_handler(self):
        """处理导入知识库文件。"""
        selected_file = filedialog.askopenfilename(
            title="选择要导入的知识库文件",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if selected_file:
            def task():
                self.disable_button(self.btn_import_knowledge)
                try:
                    self.log(f"开始导入知识库文件: {selected_file}")
                    import_knowledge_file(
                        api_key=self.api_key_var.get().strip(),
                        base_url=self.base_url_var.get().strip(),
                        file_path=selected_file
                    )
                    self.log("✅ 知识库文件导入完成。")
                except Exception as e:
                    self.log(f"❌ 导入知识库时出错: {e}")
                finally:
                    self.enable_button(self.btn_import_knowledge)

            thread = threading.Thread(target=task)
            thread.start()

    def clear_vectorstore_handler(self):
        """
        清空向量库按钮：弹出二次确认，若确认则执行 clear_vector_store()。
        """
        def confirmed_clear():
            second_confirm = messagebox.askyesno("二次确认", "你确定真的要删除所有向量数据吗？此操作不可恢复！")
            if second_confirm:
                clear_vector_store()
                self.log("已清空向量库。")

        first_confirm = messagebox.askyesno("警告", "确定要清空本地向量库吗？此操作不可恢复！")
        if first_confirm:
            confirmed_clear()

    # =========== 新增：在 UI 中查看当前剧情要点 =============
    def show_plot_arcs_ui(self):
        filepath = self.filepath_var.get().strip()
        plot_arcs_file = os.path.join(filepath, "plot_arcs.txt")
        if not os.path.exists(plot_arcs_file):
            messagebox.showinfo("剧情要点", "当前还未生成任何剧情要点或未解决冲突。")
            return
        arcs_text = read_file(plot_arcs_file).strip()
        if not arcs_text:
            arcs_text = "当前没有记录的剧情要点或冲突。"
        # 弹出一个简单的弹窗显示
        top = tk.Toplevel(self.master)
        top.title("剧情要点/未解决冲突")
        text_area = scrolledtext.ScrolledText(top, width=60, height=20)
        text_area.pack(fill="both", expand=True)
        text_area.insert(tk.END, arcs_text)
        text_area.config(state=tk.DISABLED)

    def get_llm_model(self, model_name, api_key, base_url, temperature):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature
        )
