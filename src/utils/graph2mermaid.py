import os
import inspect
import re

def create_mermaid(graph, path=None):
    if path is None:
        caller_frame = inspect.currentframe().f_back
        caller_file = caller_frame.f_code.co_filename
        caller_dir = os.path.dirname(caller_file)
        path = os.path.join(caller_dir, "graph.mmd")
    # 生成 Mermaid 代碼
    mermaid_code = graph.get_graph().draw_mermaid()
    # 移除現有的 config 區塊
    mermaid_code = re.sub(r'---\nconfig:\n.*?\n---\n', '', mermaid_code, flags=re.DOTALL)
    # 添加統一的 config 配置
    config_header = """---
config:
  theme: default
  flowchart:
    curve: linear
---
"""
    full_mermaid_code = config_header + mermaid_code
    # 保存到文件
    with open(path, "w") as f:
        f.write(full_mermaid_code)
    print(f"Graph 的 Mermaid 代碼已保存為 {path}")