# -*- coding: utf-8 -*-
"""
Image downloading and composing functionality for the PicSearch plugin.
"""
import asyncio
import io
import math
import random
import ssl
from typing import List, Optional

import aiohttp
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from astrbot.api import logger

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
]
DEFAULT_HEADERS = {'User-Agent': random.choice(USER_AGENTS)}


async def _download_image(url: str, retries: int = 2) -> Optional[bytes]:
    """Asynchronously downloads a single image with retries."""
    last_exception = None
    # Create an SSL context that does not verify certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    for attempt in range(retries):
        try:
            # Set a reasonable timeout for each attempt
            timeout = aiohttp.ClientTimeout(total=15, connect=5, sock_read=10)
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(headers=DEFAULT_HEADERS, timeout=timeout, connector=connector) as session:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    return await resp.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_exception = e
            if isinstance(e, aiohttp.ClientResponseError) and e.status in [403, 404]:
                logger.error(f"Download failed with permanent error {e.status} for {url}, skipping retries.")
                return None
            if isinstance(e, aiohttp.ClientSSLError):
                logger.error(f"Download failed with SSL error for {url}, skipping retries: {e}")
                return None
            
            logger.warning(f"PicSearch Composer: Download attempt {attempt + 1}/{retries} failed for {url}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(1)  # Wait before retrying
    
    logger.error(f"PicSearch Composer: All {retries} download attempts failed for {url}. Last error: {last_exception}")
    return None


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """Attempts to load a preferred font, with fallbacks."""
    font_names = ["Arial.ttf", "msyh.ttc", "simsun.ttc"] # Common on Win/some Linux
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size)
        except IOError:
            continue
    logger.warning("PicSearch Composer: Could not find preferred fonts. Falling back to default.")
    return ImageFont.load_default()

def _create_collage_sync(image_bytes_list: List[bytes], original_urls: List[str]) -> (Optional[bytes], List[str]):
    """
    Synchronous function to create a collage from image bytes.
    Designed to be run in an executor to avoid blocking the event loop.
    """
    successful_images, successful_urls = [], []
    tile_size = 256
    for i, img_bytes in enumerate(image_bytes_list):
        if img_bytes:
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
                successful_images.append(img)
                successful_urls.append(original_urls[i])
            except (IOError, UnidentifiedImageError) as e:
                logger.warning(f"PicSearch Composer: Skipping image from {original_urls[i]} due to processing error: {e}")
                continue

    if not successful_images:
        logger.error("PicSearch Composer: No images could be successfully processed for the collage.")
        return None, []

    columns = 4
    rows = math.ceil(len(successful_images) / columns)
    collage = Image.new('RGB', (columns * tile_size, rows * tile_size), (255, 255, 255))
    draw = ImageDraw.Draw(collage)
    font = _get_font(24)

    for i, img in enumerate(successful_images):
        row, col = i // columns, i % columns
        x_offset, y_offset = col * tile_size, row * tile_size
        collage.paste(img, (x_offset, y_offset))
        
        label = str(i + 1)
        # Simple label background
        bg_box = [x_offset + 5, y_offset + 5, x_offset + 30, y_offset + 30]
        draw.rectangle(bg_box, fill="black")
        draw.text((x_offset + 8, y_offset + 8), label, fill="white", font=font)

    buffer = io.BytesIO()
    collage.save(buffer, format="PNG")
    return buffer.getvalue(), successful_urls

async def create_collage(image_urls: List[str]) -> (Optional[bytes], List[str]):
    """
    Asynchronously downloads images and creates a collage using a non-blocking executor
    for image processing.
    """
    tasks = [_download_image(url) for url in image_urls]
    results = await asyncio.gather(*tasks)

    # Filter out failed downloads
    successful_downloads = [(img_bytes, url) for img_bytes, url in zip(results, image_urls) if img_bytes]
    if not successful_downloads:
        logger.error("PicSearch Composer: No images could be successfully downloaded for the collage.")
        return None, []
        
    image_bytes_list, original_urls = zip(*successful_downloads)

    loop = asyncio.get_running_loop()
    # Run the CPU-bound image processing in a thread pool executor
    collage_bytes, successful_urls = await loop.run_in_executor(
        None,
        _create_collage_sync,
        list(image_bytes_list),
        list(original_urls)
    )
    
    return collage_bytes, successful_urls