# -*- coding: utf-8 -*-
"""
VLM interaction functionality for the PicSearch plugin.
"""
import base64
import json
import re
import asyncio
from typing import List
 
from astrbot.api import logger
from astrbot.api.provider import Provider
 
 
async def select_from_collage(image_bytes: bytes, prompt: str, vlm_provider: Provider) -> List[int]:
    """
    Calls the VLM provider to select images from a collage using a robust,
    JSON-based prompt with regex fallback.
 
    Args:
        image_bytes (bytes): The collage image bytes.
        prompt (str): The user's prompt describing the desired image.
        vlm_provider (Provider): The AstrBot provider instance to use.
 
    Returns:
        List[int]: A list of 1-based indices of the selected images.
    """
    base64_str = base64.b64encode(image_bytes).decode('utf-8')
    image_url = f"base64://{base64_str}"
 
    prompt_template = f"""
    This is a grid of images, each with a numeric label. Please observe each labeled image carefully.
    Based on the following description: '{prompt}', identify all matching images.
    
    Your response MUST be a JSON object containing a single key "selected_indices".
    The value should be a list of the numeric labels of all matching images.
    Do not include any other text, explanations, or markdown formatting outside of the JSON object.
    
    Example of a valid response:
    {{
      "selected_indices": [1, 5, 8]
    }}
    """
    
    retries = 3
    for attempt in range(retries):
        try:
            response = await vlm_provider.text_chat(prompt=prompt_template, image_urls=[image_url])
            result = response.result_chain.get_plain_text()
            logger.debug(f"PicSearch VLM: Raw response received: '{result}'")
 
            # 1. Attempt to parse structured JSON response
            try:
                # Extract JSON part from markdown code blocks if present
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    data = json.loads(json_str)
                    selected = data.get("selected_indices")
                    if isinstance(selected, list):
                        logger.info(f"PicSearch VLM: Successfully parsed JSON response. Indices: {selected}")
                        # Ensure all elements are integers
                        return [int(n) for n in selected if isinstance(n, (int, str)) and str(n).isdigit()]
                
                # If we reach here, JSON parsing failed or format was incorrect
                logger.warning("PicSearch VLM: Failed to parse JSON, falling back to regex.")
 
            except (json.JSONDecodeError, AttributeError):
                logger.warning(f"PicSearch VLM: JSONDecodeError or invalid structure, falling back to regex. Raw: '{result}'")
 
            # 2. Fallback: Use regex to find all numbers in the raw response
            numbers = re.findall(r'\d+', result)
            if not numbers:
                logger.warning(f"PicSearch VLM: Fallback regex also found no numbers. Raw response: '{result}'")
                # This is a valid (empty) response, no need to retry
                return []
            
            logger.info(f"PicSearch VLM: Successfully extracted indices using fallback regex. Indices: {numbers}")
            return [int(n) for n in numbers]
 
        except Exception as e:
            logger.warning(f"PicSearch VLM: Attempt {attempt + 1}/{retries} failed: {e}", exc_info=True)
            if attempt < retries - 1:
                await asyncio.sleep(2)
    
    logger.error("PicSearch VLM: All attempts to call VLM provider failed.")
    return []