import time
# -*- coding: utf-8 -*-
"""
vlm_selector.py

这个脚本通过一个兼容OpenAI的API来分析带标签的图片，并根据文本描述找出最匹配的图片区域。
这允许用户连接到自托管的视觉语言模型（VLM）服务。

依赖:
- openai: 通过 `pip install openai` 安装。
- Pillow: 通过 `pip install Pillow` 安装。
- python-dotenv: 通过 `pip install python-dotenv` 安装。

环境配置:
1. 将 `.env.example` 文件复制为 `.env`。
2. 在 `.env` 文件中填入你的配置信息：
   - OPENAI_API_KEY: 你的API密钥。
   - OPENAI_API_BASE: 你的自托管API端点URL。
   - OPENAI_MODEL_NAME: 你要使用的模型名称。

用法:
python vlm_selector.py <图片路径> "<文本描述>"

示例:
python vlm_selector.py labeled_screenshot.png "一只正在打哈欠的猫"
"""

import os
import sys
import base64
import io
import time
from typing import Optional

from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()

def encode_pil_image_to_base64(image: Image.Image) -> str:
    """将Pillow图片对象编码为Base64字符串。"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def encode_image_file_to_base64(image_path: str) -> str:
    """将图片文件编码为Base64字符串。"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"错误：图片文件未找到于 '{image_path}'。", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误：编码图片时出错: {e}", file=sys.stderr)
        sys.exit(1)

def select_best_image(image: Image.Image, prompt: str) -> Optional[int]:
    """
    调用VLM API，根据文本提示从给定的Pillow图片中选择最佳区域。

    Args:
        image (Image.Image): 带有数字标签的Pillow图片对象。
        prompt (str): 用于描述所需图片的文本提示。

    Returns:
        Optional[int]: VLM选择的图片数字标签。如果出错或无法解析，则返回None。
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    model_name = os.getenv("OPENAI_MODEL_NAME")

    if not all([api_key, base_url, model_name]):
        print("错误：请确保 OPENAI_API_KEY, OPENAI_API_BASE, 和 OPENAI_MODEL_NAME 环境变量都已设置。", file=sys.stderr)
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)
    base64_image = encode_pil_image_to_base64(image)
    image_url = f"data:image/png;base64,{base64_image}"

    prompt_template = f"""
    这是一张由多张图片合成的网格图，每张图片都有一个数字标签。请仔细观察每张带标签的图片。
    根据以下描述：'{prompt}'，找出最匹配的图片。
    你的回答必须**仅仅**是那张最匹配图片的数字标签，不要包含任何其他文字、解释或标点符号。
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_template},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                max_tokens=10,
            )

            result = response.choices[0].message.content.strip()
            return int(result)

        except (ValueError, TypeError) as e:
            print(f"错误：无法将VLM的响应 '{result}' 解析为数字。尝试次数 {attempt + 1}/{max_retries}。", file=sys.stderr)
        except Exception as e:
            print(f"调用API时发生错误: {e}。尝试次数 {attempt + 1}/{max_retries}。", file=sys.stderr)
        
        if attempt < max_retries - 1:
            time.sleep(2) # 等待2秒后重试

    print("错误：经过多次尝试后，仍未能从VLM获取有效响应。", file=sys.stderr)
    return None

def call_vlm_api(image_path: str, text_prompt: str) -> Optional[str]:
    """
    【兼容旧版】通过文件路径调用VLM API。
    打开图片文件，然后调用新的 `select_best_image` 函数。
    """
    try:
        with Image.open(image_path) as img:
            result = select_best_image(img, text_prompt)
            return str(result) if result is not None else None
    except FileNotFoundError:
        print(f"错误：图片文件未找到于 '{image_path}'。", file=sys.stderr)
        return None
    except Exception as e:
        print(f"处理图片时出错: {e}", file=sys.stderr)
        return None

def main():
    """
    主函数，处理命令行参数并驱动脚本流程。
    """
    if len(sys.argv) != 3:
        print("用法: python vlm_selector.py <图片路径> \"<文本描述>\"", file=sys.stderr)
        sys.exit(1)

    image_file_path = sys.argv[1]
    user_text_prompt = sys.argv[2]

    selected_label = call_vlm_api(image_file_path, user_text_prompt)

    if selected_label is not None:
        print(selected_label)
    else:
        print("未能从VLM获取有效选择。", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()