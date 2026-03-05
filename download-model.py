# download_model.py
from huggingface_hub import snapshot_download
import os

model_dir = "./models/chatglm3-6b"
os.makedirs(model_dir, exist_ok=True)

print("开始下载 ChatGLM-6B 模型...")

snapshot_download(
    repo_id="THUDM/chatglm3-6b",
    local_dir=model_dir,
    local_dir_use_symlinks=False,
)

print(f"模型已下载到：{model_dir}")