# -*- coding: utf-8 -*-
"""
VLM interaction functionality for the PicSearch plugin.
"""
import base64
import re
from typing import List

from astrbot.api import logger
from astrbot.api.provider import Provider


async def select_from_collage(image_bytes: bytes, prompt: str, vlm_provider: Provider) -> List[int]:
    """
    Calls the VLM provider to select images from a collage.

    Args:
        image_bytes (bytes): The collage image bytes.
        prompt (str): The user's prompt describing the desired image.
        vlm_provider (Provider): The AstrBot provider instance to use.

    Returns:
        List[int]: A list of 1-based indices of the selected images.
    """
    base64_str = base64.b64encode(image_bytes).decode('utf-8')
    # Use the base64 scheme supported by the framework
    image_url = f"base64://{base64_str}"

    prompt_template = f"""
    This is a grid of images, each with a numeric label. Please observe each labeled image carefully.
    Based on the following description: '{prompt}', identify all matching images.
    Your response must be **only** the numeric labels of the matching images, separated by commas,
    with no other text, explanations, or punctuation. For example: 1,5,8
    """
    try:
        response = await vlm_provider.text_chat(prompt=prompt_template, image_urls=[image_url])
        result = response.result_chain.get_plain_text()
        
        # Find all numbers in the response string
        numbers = re.findall(r'\d+', result)
        if not numbers:
            logger.warning(f"PicSearch VLM: Response contained no numbers. Raw response: '{result}'")
            return []
            
        return [int(n) for n in numbers]
    except Exception as e:
        logger.error(f"PicSearch VLM: An error occurred during text_chat call: {e}", exc_info=True)
        return []