# -*- coding: utf-8 -*-
"""
Image downloading and composing functionality for the PicSearch plugin.
"""
import asyncio
import io
import math
import random
from typing import List, Optional

import aiohttp
from PIL import Image, ImageDraw, ImageFont

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
]
DEFAULT_HEADERS = {'User-Agent': random.choice(USER_AGENTS)}


async def _download_image(url: str) -> Optional[bytes]:
    """Asynchronously downloads a single image."""
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(headers=DEFAULT_HEADERS, timeout=timeout) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                return await resp.read()
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None


async def create_collage(image_urls: List[str]) -> (Optional[bytes], List[str]):
    """
    Asynchronously downloads images and creates a collage.

    Args:
        image_urls (List[str]): A list of URLs to download.

    Returns:
        A tuple containing:
        - bytes: The collage image in PNG format.
        - List[str]: A list of URLs that were successfully downloaded and included.
    """
    tasks = [_download_image(url) for url in image_urls]
    results = await asyncio.gather(*tasks)

    successful_images, successful_urls = [], []
    tile_size = 256
    for i, img_bytes in enumerate(results):
        if img_bytes:
            try:
                # Pillow operations are synchronous but fast enough for this context.
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
                successful_images.append(img)
                successful_urls.append(image_urls[i])
            except Exception:  # Ignore images that fail to process
                continue

    if not successful_images:
        return None, []

    columns = 4
    rows = math.ceil(len(successful_images) / columns)
    collage = Image.new('RGB', (columns * tile_size, rows * tile_size), (255, 255, 255))
    draw = ImageDraw.Draw(collage)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    for i, img in enumerate(successful_images):
        row, col = i // columns, i % columns
        x_offset, y_offset = col * tile_size, row * tile_size
        collage.paste(img, (x_offset, y_offset))
        
        label = str(i + 1)
        # Simple label background
        bg_box = [x_offset + 5, y_offset + 5, x_offset + 28, y_offset + 28]
        draw.rectangle(bg_box, fill="black")
        draw.text((x_offset + 8, y_offset + 8), label, fill="white", font=font)

    buffer = io.BytesIO()
    collage.save(buffer, format="PNG")
    return buffer.getvalue(), successful_urls