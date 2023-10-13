import os
import shutil

# 源文件夹路径
source_folder = "./data/deep/test"

# 目标文件夹路径
target_folder = "./data/deep/test/A"

# 确保目标文件夹存在，如果不存在则创建它
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 列出源文件夹中以"lr.tif"结尾的文件S
for filename in os.listdir(source_folder):
    if filename.endswith("lr.tif"):
        source_file = os.path.join(source_folder, filename)
        target_file = os.path.join(target_folder, filename)

        # 移动文件到目标文件夹
        shutil.move(source_file, target_file)
        print(f"Moved {filename} to {target_folder}")
