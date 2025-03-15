import tkinter as tk
from tkinter import messagebox
import pandas as pd

from test import all_score
# 示例的本地计算得分函数
def calculate_score(text1, text2):
    # 这里可以实现具体的得分计算逻辑
    # 为了示例，简单返回两文本长度之和
    return len(text1) + len(text2)

class TableViewer:
    def __init__(self, root, file_path):
        self.root = root
        self.root.title("法律翻译评价")

        # 读取 CSV 文件
        self.data = pd.read_excel(file_path)
        self.current_index = 0

        # 创建标签用于显示表格信息
        self.label1 = tk.Label(root, text="", font=("Arial", 12))
        self.label1.pack(pady=10)
        self.label2 = tk.Label(root, text="", font=("Arial", 12))
        self.label2.pack(pady=10)

        # 创建文本框用于显示得分
        self.score_entry = tk.Entry(root, font=("Arial", 12))
        self.score_entry.pack(pady=10)

        # 创建计算得分按钮
        self.calculate_button = tk.Button(root, text="计算得分", command=self.calculate)
        self.calculate_button.pack(pady=10)

        # 创建下一条按钮
        self.next_button = tk.Button(root, text="下一条", command=self.next_entry)
        self.next_button.pack(pady=10)

        # 显示第一行信息
        self.show_current_entry()

    def show_current_entry(self):
        if self.current_index < len(self.data):
            row = self.data.iloc[self.current_index]
            col1_value = row[0]
            col2_value = row[1]
            self.label1.config(text=f"第一列信息: {col1_value}")
            self.label2.config(text=f"第二列信息: {col2_value}")
            self.score_entry.delete(0, tk.END)  # 清空得分文本框
        else:
            messagebox.showinfo("提示", "表格已处理完成")
            self.calculate_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)

    def calculate(self):
        if self.current_index < len(self.data):
            row = self.data.iloc[self.current_index]
            col1_value = row[0]
            col2_value = row[1]
            # score,term_flag,legal_flag,effect_flag = calculate_score(col1_value, col2_value)
            score,term_flag,legal_flag,effect_flag = all_score(col1_value, col2_value)
            self.score_entry.delete(0, tk.END)
            self.score_entry.insert(0, str(score))

    def next_entry(self):
        self.current_index += 1
        self.show_current_entry()

if __name__ == "__main__":
    root = tk.Tk()
    file_path = "测试语料英文-1.xlsx"  # 替换为实际的 CSV 文件路径
    app = TableViewer(root, file_path)
    root.mainloop()