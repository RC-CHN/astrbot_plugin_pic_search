# -*- coding: utf-8 -*-
"""
This module provides functionalities for creating image collages from URLs
and processing them in batches to select a single best image using a VLM.

Dependencies:
- Pillow (PIL)
- requests
- tqdm (for progress bar)
"""

import io
import math
import random
import concurrent.futures
from urllib.parse import urlparse
from typing import List, Callable, Optional

import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# A list of common user agents to rotate through to avoid being blocked.
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
]

def _download_and_process_image(url: str, tile_size: int) -> Optional[Image.Image]:
    """
    Downloads and processes a single image from a URL.
    Uses a random user-agent and a dynamic referer to reduce blocking.

    Args:
        url (str): The URL of the image to download.
        tile_size (int): The target size for the image tile.

    Returns:
        A processed PIL Image object, or None if an error occurs.
    """
    try:
        parsed_url = urlparse(url)
        referer = f"{parsed_url.scheme}://{parsed_url.netloc}/"
        
        headers = {
            'User-Agent': random.choice(USER_AGENTS),
            'Referer': referer
        }
        
        # Set a 5-second timeout for the request
        response = requests.get(url, timeout=5, headers=headers)
        response.raise_for_status()
        
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        img = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        return img
    except requests.exceptions.RequestException:
        # Silently fail on any request exception (e.g., timeout, connection error)
        return None
    except (IOError, ValueError) as e:
        # Log other potential errors like image processing issues, but still fail silently
        # print(f"Warning: Could not process image {url}. Skipping. Error: {e}")
        return None

def create_image_collage(
    image_urls: List[str],
    tile_size: int = 256,
    columns: int = 4,
    debug_filename: Optional[str] = None
) -> (Optional[Image.Image], List[str]):
    """
    Downloads images from URLs and creates a collage.

    Returns:
        A tuple containing the collage Image and a list of successfully downloaded image URLs.
        Returns (None, []) if no images could be processed.
    """
    if not image_urls:
        return None, []

    successful_images = []
    successful_urls = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_url = {executor.submit(_download_and_process_image, url, tile_size): url for url in image_urls}
        
        results = tqdm(
            concurrent.futures.as_completed(future_to_url),
            total=len(image_urls),
            desc="Downloading images..."
        )
        
        for future in results:
            url = future_to_url[future]
            img = future.result()
            if img:
                successful_images.append(img)
                successful_urls.append(url)

    if not successful_images:
        return None, []

    rows = math.ceil(len(successful_images) / columns)
    grid_width = columns * tile_size
    grid_height = rows * tile_size
    
    collage = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    draw = ImageDraw.Draw(collage)
    
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    for i, img in enumerate(successful_images):
        row = i // columns
        col = i % columns
        x_offset = col * tile_size
        y_offset = row * tile_size
        collage.paste(img, (x_offset, y_offset))
        
        # Add a numbered label
        label = str(i + 1)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Create a background for the label
        bg_padding = 5
        bg_box = [
            x_offset + bg_padding, 
            y_offset + bg_padding, 
            x_offset + text_width + bg_padding * 2, 
            y_offset + text_height + bg_padding * 2
        ]
        draw.rectangle(bg_box, fill="black")
        draw.text((x_offset + bg_padding, y_offset + bg_padding), label, fill="white", font=font)

    # If a debug filename is provided, save the collage to disk
    if debug_filename:
        try:
            collage.save(debug_filename)
            print(f"Debug collage saved to {debug_filename}")
        except IOError as e:
            print(f"Warning: Could not save debug collage to {debug_filename}. Error: {e}")

    return collage, successful_urls

def process_in_batches(
    image_urls: List[str],
    vlm_selector_func: Callable[[Image.Image, int], List[int]],
    debug: bool = False
) -> str:
    """
    Processes images in batches, using a VLM to select the best one recursively.

    Args:
        image_urls (List[str]): The initial list of image URLs.
        vlm_selector_func (Callable): A function that takes a collage image and the
                                     number of images in it, and returns a list
                                     of indices of the selected images.

    Returns:
        str: The URL of the single, final winning image.
    """
    if not image_urls:
        raise ValueError("Initial image URL list cannot be empty.")

    current_winners = image_urls
    batch_size = 16

    while len(current_winners) > 1:
        next_round_winners = []
        batch_counter = 0
        
        # Process current winners in batches
        for i in range(0, len(current_winners), batch_size):
            batch_urls = current_winners[i:i + batch_size]
            
            if not batch_urls:
                continue

            print(f"Processing a batch of {len(batch_urls)} images...")
            
            # Create a collage for the current batch
            debug_filename = f"debug_collage_batch_{batch_counter}.png" if debug else None
            collage_image, successful_urls = create_image_collage(batch_urls, debug_filename=debug_filename)
            batch_counter += 1

            if not collage_image or not successful_urls:
                print("Warning: No images were successfully processed for this batch. Skipping.")
                continue

            # Use the VLM to select the best images from the collage
            num_images_in_collage = len(successful_urls)
            selected_indices = vlm_selector_func(collage_image, num_images_in_collage)
            
            # Collect the URLs of the selected images from the list of successful ones
            for index in selected_indices:
                # The VLM returns a 1-based index, so we convert to 0-based.
                actual_index = index - 1
                if 0 <= actual_index < len(successful_urls):
                    next_round_winners.append(successful_urls[actual_index])
                else:
                    print(f"Warning: VLM returned an invalid index {index}. Ignoring.")

        if not next_round_winners:
            raise RuntimeError("VLM selection process resulted in no winners.")
            
        current_winners = next_round_winners
        print(f"Round complete. {len(current_winners)} winners moving to the next round.")

    if not current_winners:
        raise RuntimeError("The selection process concluded with no final winner.")
        
    return current_winners[0]