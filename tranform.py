import os
from pathlib import Path

def delete_images_until_150(folder_path, target=150):
    """
    删除图片直到只剩 target 张，删除方式：按顺序隔一个删一个，循环进行。
    """
    folder = Path(folder_path)

    # 检查路径
    if not folder.exists():
        print(f"❌ 文件夹不存在: {folder}")
        return
    if not folder.is_dir():
        print(f"❌ 路径不是文件夹: {folder}")
        return

    # 支持的图片格式
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    # 获取所有图片并排序
    images = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in image_exts]
    images.sort()  # 按文件名排序

    current_count = len(images)
    print(f"🔍 找到 {current_count} 张图片，目标保留 {target} 张。")

    if current_count <= target:
        print("✅ 图片数量已达标，无需删除。")
        return

    # 循环删除：每轮隔一个删一个，直到 ≤150
    while len(images) > target:
        # 本轮要删除的索引：0, 2, 4, ...（即第1、3、5...张）
        indices_to_delete = [i for i in range(0, len(images), 2)]

        # 控制删除数量，避免删过头
        if len(images) - len(indices_to_delete) < target:
            can_delete = len(images) - target
            indices_to_delete = indices_to_delete[:can_delete]

        # 倒序删除，避免索引错乱
        for idx in sorted(indices_to_delete, reverse=True):
            img_file = images[idx]
            try:
                img_file.unlink()  # 直接删除文件
                print(f"🗑️ 删除: {img_file.name}")
                images.pop(idx)
            except Exception as e:
                print(f"⚠️ 删除失败 {img_file.name}: {e}")

        print(f"📊 当前剩余图片数: {len(images)}")

        if len(images) <= target:
            break

    print(f"✅ [{folder.name}] 处理完成！最终保留 {len(images)} 张图片。\n")


def process_parent_folder(parent_folder_path, target=150):
    """
    遍历父文件夹下的所有子文件夹，并对每个子文件夹执行删除操作。
    """
    parent = Path(parent_folder_path)

    if not parent.exists():
        print(f"❌ 父文件夹不存在: {parent}")
        return
    if not parent.is_dir():
        print(f"❌ 输入的不是一个文件夹: {parent}")
        return

    # 获取所有直接子文件夹
    subfolders = [f for f in parent.iterdir() if f.is_dir()]

    if not subfolders:
        print(f"🔍 在 {parent} 中未找到任何子文件夹。")
        return

    print(f"📁 发现 {len(subfolders)} 个子文件夹，开始逐个处理...\n")

    for idx, subfolder in enumerate(subfolders, start=1):
        print(f"📌 正在处理第 {idx}/{len(subfolders)} 个文件夹: {subfolder.name}")
        delete_images_until_150(subfolder, target=target)
        print("-" * 50)

    print(f"🎉 所有子文件夹处理完成！")


# ============ 使用时修改这里 ============
if __name__ == "__main__":
    # 🔧 修改为你的大文件夹（父文件夹）路径
    PARENT_FOLDER_PATH = r"D:\part\GNM\05B\22"

    process_parent_folder(PARENT_FOLDER_PATH, target=150)