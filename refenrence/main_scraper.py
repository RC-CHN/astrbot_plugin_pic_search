# -*- coding: utf-8 -*-
"""
主应用程序脚本，整合了图片抓取、合成和VLM选择功能。

该脚本执行以下操作：
1. 使用bing_image_scraper从Bing图片搜索抓取图片URL。
2. 使用image_composer将图片分批合成为带标签的网格图。
3. 使用vlm_selector通过VLM（视觉语言模型）从合成图中选择最符合描述的图片。
4. 递归地重复此过程，直到只剩下一张最终的优胜图片。

用法:
python main_scraper.py -q "你的搜索查询" -p "你希望VLM寻找的图片的详细描述" -c <数量>
"""

import argparse
import sys
from typing import List

from PIL import Image

# 从项目模块中导入所需函数
from bing_image_scraper import scrape_images
from image_composer import process_in_batches
from vlm_selector import select_best_image

def main():
    """
    主执行函数，处理命令行参数并协调整个工作流程。
    """
    parser = argparse.ArgumentParser(
        description="抓取图片，并使用VLM根据文本提示选择最佳图片。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        required=True,
        help="用于Bing图片搜索的关键词。"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        required=True,
        help="用于VLM分析的详细文本提示，描述你想要的图片特征。"
    )
    parser.add_argument(
        "-c", "--count",
        type=int,
        default=30,
        help="期望抓取的图片数量（默认为30）。"
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="如果设置，将中间生成的拼贴图保存到磁盘。"
    )
    args = parser.parse_args()

    print(f"开始任务：使用查询 '{args.query}' 抓取 {args.count} 张图片，并用提示 '{args.prompt}' 进行筛选。")

    # 步骤 1: 抓取图片URL
    print("\n--- 步骤 1: 正在从Bing抓取图片URL... ---")
    try:
        image_urls = scrape_images(args.query, args.count)
        if not image_urls:
            print("未能抓取到任何图片URL，程序终止。")
            sys.exit(1)
        print(f"成功抓取到 {len(image_urls)} 个图片URL。")
    except Exception as e:
        print(f"抓取图片时发生错误: {e}")
        sys.exit(1)

    # 步骤 2: 定义VLM选择器函数闭包
    # 这个函数将作为回调传递给process_in_batches
    def vlm_selector_callback(collage_image: Image.Image, num_images: int) -> List[int]:
        """
        一个包装器，用于调用VLM选择函数并处理其输出。
        
        Args:
            collage_image (Image.Image): 包含多个候选图片的合成图。
            num_images (int): 合成图中的图片数量。

        Returns:
            List[int]: VLM选择的图片的1-based索引列表。
        """
        print(f"正在调用VLM来分析一个包含 {num_images} 张图片的合成图...")
        selected_number = select_best_image(collage_image, args.prompt)
        
        if selected_number is not None:
            if 1 <= selected_number <= num_images:
                print(f"VLM选择了图片 #{selected_number}。")
                # 直接返回VLM选择的1-based数字
                return [selected_number]
            else:
                print(f"警告：VLM返回了一个无效的数字 {selected_number}，该数字超出了当前拼贴图的范围（共 {num_images} 张图）。")
                return []
        else:
            print("警告：VLM未能在此批次中做出有效选择。")
            return []

    # 步骤 3: 处理图片批次并找出最终胜利者
    print("\n--- 步骤 2: 开始通过VLM进行分批选择... ---")
    try:
        final_winner_url = process_in_batches(image_urls, vlm_selector_callback, debug=args.debug)
        
        # 步骤 4: 打印最终结果
        print("\n--- 任务完成 ---")
        print(f"最终选择的图片URL是: {final_winner_url}")

    except (ValueError, RuntimeError) as e:
        print(f"\n处理过程中发生错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n发生未知错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()