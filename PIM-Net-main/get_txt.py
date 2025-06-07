import os

def generate_dataset_txt_files(base_data_path="UltraEdit/", target_extension=".png"):
    """
    为数据集生成 train.txt, test.txt, 和 val.txt 文件。
    这些文件将包含在对应 source_image 目录中找到的图像的基础文件名（不含扩展名）。

    Args:
        base_data_path (str): 数据集的根路径 (例如, 'UltraEdit/')。
                                此路径应指向包含 train, test, val 子目录的文件夹。
        target_extension (str): 要查找的目标图像文件扩展名 (例如, '.png')。
                                 确保这与 data_util.py 中加载图像时使用的扩展名一致。
    """
    modes = ["train", "test", "val"]  # 数据集的模式

    print(f"开始生成 .txt 文件，目标数据集根目录: {os.path.abspath(base_data_path)}")
    print(f"将查找并记录具有扩展名 '{target_extension}' 的图像文件。")
    print("-" * 50)

    if not os.path.isdir(base_data_path):
        print(f"错误：指定的数据集根目录 '{os.path.abspath(base_data_path)}' 不存在。请检查路径。")
        return

    for mode in modes:
        # 构建到 source_image 文件夹的路径
        source_image_dir = os.path.join(base_data_path, mode, "source_image")
        # 构建输出 .txt 文件的路径
        output_txt_file = os.path.join(base_data_path, mode + ".txt")

        print(f"处理模式: {mode}")

        if not os.path.isdir(source_image_dir):
            print(f"  警告：子目录 '{os.path.abspath(source_image_dir)}' 不存在，将跳过生成 '{mode}.txt'。")
            print("-" * 50)
            continue

        image_basenames = []
        try:
            print(f"  正在扫描目录: {os.path.abspath(source_image_dir)}")
            files_in_dir = os.listdir(source_image_dir)

            if not files_in_dir:
                print(f"  警告：目录 '{os.path.abspath(source_image_dir)}' 为空。")

            for filename in files_in_dir:
                basename, ext = os.path.splitext(filename)
                if ext.lower() == target_extension.lower():
                    image_basenames.append(basename)
                # else:
                #     print(f"    跳过文件: {filename} (扩展名 {ext} 与目标 {target_extension} 不匹配)")

        except OSError as e:
            print(f"  错误：无法读取目录 '{os.path.abspath(source_image_dir)}' 下的文件：{e}")
            print("-" * 50)
            continue

        if not image_basenames:
            print(f"  警告：在 '{os.path.abspath(source_image_dir)}' 中未找到扩展名为 '{target_extension}' 的图像文件。")
            print(f"  因此，生成的 '{os.path.abspath(output_txt_file)}' 将为空文件。")
        else:
            print(f"  在 '{os.path.abspath(source_image_dir)}' 中找到 {len(image_basenames)} 个 '{target_extension}' 文件。")

        # 将收集到的基础文件名写入 .txt 文件
        try:
            # 使用 'utf-8' 编码以支持中文等特殊字符的文件名
            with open(output_txt_file, 'w', encoding='utf-8') as f:
                # 对文件名进行排序，以确保每次生成的文件内容顺序一致
                for basename in sorted(image_basenames):
                    f.write(basename + "\n")
            print(f"  成功生成/更新文件: {os.path.abspath(output_txt_file)}，包含 {len(image_basenames)} 个条目。")
        except IOError as e:
            print(f"  错误：无法写入文件 '{os.path.abspath(output_txt_file)}'：{e}")
        print("-" * 50)

if __name__ == "__main__":
    # --- 用户配置区域 ---
    # 请根据您的 'UltraEdit' 文件夹的实际位置来设置此路径。
    # 例如:
    # 1. 如果此脚本文件与 'UltraEdit' 文件夹在同一目录中:
    #    dataset_directory = "UltraEdit/"
    # 2. 如果 'UltraEdit' 文件夹在其他特定位置:
    #    dataset_directory = "/mnt/d/BaiduDownload/data/UltraEdit/"
    #
    # 根据您之前提供的 `tree` 命令的上下文，您的 'UltraEdit' 文件夹位于 '/mnt/d/BaiduDownload/data/' 下。
    # 如果您将此 Python 脚本保存在 '/mnt/d/BaiduDownload/data/' 目录中，则下面的路径是正确的。
    dataset_directory = "UltraEdit/"

    # 确保此扩展名与您的 data_util.py 脚本中 __getitem__ 方法加载图像时所期望的扩展名一致。
    # 在我们之前修改的 data_util.py 中，使用的是 name + '.png'。
    image_file_extension = ".png"
    # --- 用户配置区域结束 ---

    print("="*10 + " 开始执行脚本 " + "="*10)
    generate_dataset_txt_files(base_data_path=dataset_directory, target_extension=image_file_extension)
    print("\n脚本执行完毕。")
    print(f"请检查 '{os.path.abspath(dataset_directory)}' 目录下是否已生成或更新 train.txt, test.txt, 和 val.txt 文件。")