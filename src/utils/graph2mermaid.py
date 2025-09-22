import os
import inspect

def create_mermaid(graph, path=None):
    if path is None:
        caller_frame = inspect.currentframe().f_back
        caller_file = caller_frame.f_code.co_filename
        caller_dir = os.path.dirname(caller_file)
        path = os.path.join(caller_dir, "graph.mmd")
    # 生成 Mermaid 代碼
    mermaid_code = graph.get_graph().draw_mermaid()
    # 保存到文件
    with open(path, "w") as f:
        f.write(mermaid_code)
    print(f"Graph 的 Mermaid 代碼已保存為 {path}")