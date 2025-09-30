import os
import random


def rename_images_in_subfolders(root_folder):
    """
    对root_folder下的每个子文件夹中的jpg文件进行重命名。
    每个子文件夹内的jpg文件将被命名为从1到150的连续数字，
    例如：1.jpg, 2.jpg, ..., 150.jpg。
    """
    # 遍历根文件夹下的所有子文件夹
    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)

        # 确保是目录
        if os.path.isdir(subdir_path):
            # 获取子目录下所有的jpg文件
            files = [f for f in os.listdir(subdir_path) if f.lower().endswith('.jpg')]

            # 如果子目录下的jpg文件数量不是150，则跳过该目录
            if len(files) != 150:
                print(f"警告: 子文件夹 {subdir} 包含 {len(files)} 张图片而不是150张。此文件夹未被处理。")
                continue
            # 重命名jpg文件
            for idx, old_name in enumerate(files, start=1):
                old_path = os.path.join(subdir_path, old_name)
                new_name = f"{idx}.jpg"
                new_path = os.path.join(subdir_path, new_name)

                # 执行重命名操作
                os.rename(old_path, new_path)
                print(f"Renamed: {old_name} -> {new_name} in folder {subdir}")


if __name__ == "__main__":
    # 设置你的根文件夹路径（替换为实际路径）
    root_folder = r"D:\part\GNM\data\1"
    rename_images_in_subfolders(root_folder)
    print("批量重命名完成！")