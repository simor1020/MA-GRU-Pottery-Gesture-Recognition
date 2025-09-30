import os
import shutil


def rename_images_in_subfolders(root_folder):
    """
    对root_folder下的每个子文件夹中的jpg文件进行重命名。
    每个子文件夹内的jpg文件将被命名为从1到150的连续数字。

    如果图片不足150张，则用最后一张图片复制补齐至150张。
    """
    # 遍历根文件夹下的所有子文件夹
    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)

        # 确保是目录
        if not os.path.isdir(subdir_path):
            continue

        # 获取子目录下所有.jpg文件（不区分大小写），并排序
        files = sorted([f for f in os.listdir(subdir_path) if f.lower().endswith('.jpg')])

        num_files = len(files)

        if num_files == 150:
            print(f"子文件夹 {subdir} 正好有150张图片，开始重命名。")
        elif num_files < 150:
            missing_count = 150 - num_files
            print(f"子文件夹 {subdir} 只有 {num_files} 张图片，正在用最后一张图片补齐 {missing_count} 张...")

            # 获取最后一张图片的文件名（排序后最后一个）
            last_image = files[-1]
            last_image_path = os.path.join(subdir_path, last_image)

            # 复制最后一张图片，直到总数达到150
            for i in range(1, missing_count + 1):
                new_filename = f"temp_duplicate_{i}.jpg"  # 使用临时名字避免冲突
                new_path = os.path.join(subdir_path, new_filename)
                shutil.copy2(last_image_path, new_path)  # 复制元数据和内容
                files.append(new_filename)  # 添加到列表中参与后续重命名

            # 重新获取并排序文件列表（确保新文件也被包含）
            files = sorted([f for f in os.listdir(subdir_path) if f.lower().endswith('.jpg')])
        else:
            print(f"警告: 子文件夹 {subdir} 有 {num_files} 张图片（超过150），跳过处理。")
            continue

        # 现在files应该正好有150个jpg文件，开始重命名
        for idx, old_name in enumerate(files, start=1):
            new_name = f"{idx}.jpg"
            old_path = os.path.join(subdir_path, old_name)
            new_path = os.path.join(subdir_path, new_name)

            if old_name != new_name:
                os.rename(old_path, new_path)
                # 可选：只打印实际改名的操作
                # print(f"Renamed: {old_name} -> {new_name}")

        print(f"✅ 子文件夹 {subdir} 重命名完成，共处理150张图片。")


if __name__ == "__main__":
    # 设置你的根文件夹路径（替换为实际路径）
    root_folder = r"D:\part\GNM\data\1"

    if not os.path.exists(root_folder):
        print(f"错误：指定的路径不存在：{root_folder}")
    else:
        rename_images_in_subfolders(root_folder)
        print("🎉 批量重命名与补全操作全部完成！")