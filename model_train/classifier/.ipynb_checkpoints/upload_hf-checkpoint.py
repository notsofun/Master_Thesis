from huggingface_hub import HfApi
import os

# ================= 配置区 =================
# 你的 HF 用户名
USER_NAME = "Zhidian2025" 
# 仓库名称（会自动创建，如果是私有的记得在官网设为 Private）
REPO_NAME = "Master-Thesis-Models" 
# 本地模型文件的完整路径
LOCAL_FILE_PATH = "model_train/classifier/Chinese/best_multitask_model.pt"
# 上传到 HF 仓库后叫什么名字
REMOTE_FILE_NAME = "Thu-Chinese-hate-v1.pt"
# ==========================================

def upload_model():
    api = HfApi()
    
    # 构造完整的仓库 ID
    repo_id = f"{USER_NAME}/{REPO_NAME}"
    
    print(f"正在准备上传至: {repo_id}...")
    
    try:
        # 如果仓库不存在会自动创建
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)
        
        # 执行上传
        api.upload_file(
            path_or_fileobj=LOCAL_FILE_PATH,
            path_in_repo=REMOTE_FILE_NAME,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload model checkpoint: {REMOTE_FILE_NAME}"
        )
        
        print(f"✅ 上传成功！")
        print(f"你的模型地址: https://huggingface.co/{repo_id}/blob/main/{REMOTE_FILE_NAME}")
        
    except Exception as e:
        print(f"❌ 上传失败: {e}")

if __name__ == "__main__":
    if os.path.exists(LOCAL_FILE_PATH):
        upload_model()
    else:
        print(f"错误: 找不到本地文件 {LOCAL_FILE_PATH}")