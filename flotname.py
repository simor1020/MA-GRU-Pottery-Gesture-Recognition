import os
from pathlib import Path


def rename_subfolders_sequentially(parent_folder_path):
    """
    将父文件夹下的所有子文件夹按顺序重命名为 1, 2, 3, ...
    """
    parent = Path(parent_folder_path)

    # 检查路径是否存在且为目录
    if not parent.exists():
        print(f"❌ 错误：路径不存在：{parent}")
        return
    if not parent.is_dir():
        print(f"❌ 错误：路径不是一个文件夹：{parent}")
        return

    # 获取所有子文件夹（只取直接子级）
    subfolders = [f for f in parent.iterdir() if f.is_dir()]

    if not subfolders:
        print(f"🔍 在 {parent} 中未找到任何子文件夹。")
        return

    # 按文件夹名称排序（确保顺序一致，也可以改为按创建时间排序）
    subfolders.sort(key=lambda x: x.name)  # 按名称字母顺序排序
    # 可选：按创建时间排序 → subfolders.sort(key=lambda x: x.stat().st_ctime)

    print(f"📁 在 '{parent.name}' 中发现 {len(subfolders)} 个子文件夹，开始重命名...\n")

    # 重命名
    for idx, folder in enumerate(subfolders, start=1):
        new_name = str(idx)
        new_path = folder.parent / new_name

        if folder.name == new_name:
            print(f"✅ 跳过（已是目标名）: {folder.name}")
            continue

        try:
            folder.rename(new_path)
            print(f"✅ '{folder.name}' → '{new_name}'")
        except PermissionError:
            print(f"❌ 权限错误：无法重命名 '{folder.name}'，可能正在被使用。")
        except FileExistsError:
            print(f"❌ 冲突：已存在文件夹 '{new_name}'，跳过。")
        except Exception as e:
            print(f"❌ 未知错误：{folder.name} → {e}")

    print(f"\n🎉 重命名完成！共处理 {len(subfolders)} 个子文件夹。")


# ============ 使用时修改这里 ============
if __name__ == "__main__":
    # 🔧 修改为你的大文件夹路径
    PARENT_FOLDER_PATH = r"D:\part\GNM\dataset\10"  # <-- 替换为你的实际路径

    rename_subfolders_sequentially(PARENT_FOLDER_PATH)