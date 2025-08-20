import argparse
import subprocess
import sys
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()

def run_image_search(query):
    """
    运行图像搜索脚本 (image_search_annotator.py).

    Args:
        query (str): 图像搜索的关键词.

    Returns:
        bool: 如果脚本成功执行则返回 True, 否则返回 False.
    """
    print(f"正在使用查询 '{query}' 搜索图像...")
    try:
        # 调用 image_search_annotator.py 脚本
        # 我们只关心它是否成功执行并创建了 labeled_screenshot.png
        subprocess.run(
            [sys.executable, "image_search_annotator.py", query],
            check=True,
            capture_output=True,
            text=True
        )
        print("图像搜索和标注完成。")
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: 运行 image_search_annotator.py 失败。", file=sys.stderr)
        print(f"返回码: {e.returncode}", file=sys.stderr)
        print(f"输出: {e.stdout}", file=sys.stderr)
        print(f"错误输出: {e.stderr}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("错误: 'image_search_annotator.py' 未找到。请确保它在当前目录中。", file=sys.stderr)
        return False


def run_vlm_selector(prompt):
    """
    运行 VLM 选择器脚本 (vlm_selector.py).

    Args:
        prompt (str): 用于 VLM 分析的文本提示.

    Returns:
        str: VLM 选择的最佳图片的数字标签，如果失败则返回 None.
    """
    print(f"正在使用提示 '{prompt}' 运行 VLM 选择器...")
    image_path = "labeled_screenshot.png"
    try:
        # 调用 vlm_selector.py 脚本并捕获其标准输出
        result = subprocess.run(
            [sys.executable, "vlm_selector.py", image_path, prompt],
            check=True,
            capture_output=True,
            text=True,
            encoding='gbk' # 在Windows上使用GBK编码以避免解码错误
        )
        # 移除输出中可能存在的多余空白字符
        selected_label = result.stdout.strip()
        print(f"VLM 选择器完成，输出为: '{selected_label}'")
        return selected_label
    except subprocess.CalledProcessError as e:
        print(f"错误: 运行 vlm_selector.py 失败。", file=sys.stderr)
        print(f"返回码: {e.returncode}", file=sys.stderr)
        print(f"输出: {e.stdout}", file=sys.stderr)
        print(f"错误输出: {e.stderr}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("错误: 'vlm_selector.py' 或 'labeled_screenshot.png' 未找到。", file=sys.stderr)
        return None

def main():
    """
    主函数，用于协调图像搜索和 VLM 选择过程。
    """
    # 1. 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="使用VLM从图像搜索结果中选择最佳图片。")
    parser.add_argument("-q", "--query", type=str, required=True, help="用于图像搜索的关键词。")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="用于VLM分析的文本提示。")

    args = parser.parse_args()

    # 2. 工作流程
    # 步骤 1: 运行图像搜索和标注脚本
    if not run_image_search(args.query):
        sys.exit(1) # 如果图像搜索失败，则退出

    # 步骤 2: 运行 VLM 选择器脚本
    selected_image_label = run_vlm_selector(args.prompt)

    # 步骤 3 & 4: 捕获并打印最终结果
    if selected_image_label:
        print("\n=====================================")
        print(f"VLM选择的图片是: {selected_image_label}")
        print("=====================================")
    else:
        print("\n无法从VLM选择器获取结果。", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()