# -*- coding: utf-8 -*-
"""
Image scraping functionality for the PicSearch plugin.
"""
import asyncio
import json
import random
from typing import List

import aiohttp
from bs4 import BeautifulSoup

from astrbot.api import logger

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
]
DEFAULT_HEADERS = {'User-Agent': random.choice(USER_AGENTS)}


async def scrape_image_urls(query: str, count: int) -> List[str]:
    """
    Asynchronously scrapes image URLs from Bing.

    Args:
        query (str): The search term.
        count (int): The desired number of images.

    Returns:
        List[str]: A list of direct URLs to the images.
    """
    urls, seen = [], set()
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(headers=DEFAULT_HEADERS, timeout=timeout) as session:
        first = 0
        while len(urls) < count:
            params = {"q": query, "first": first}
            try:
                async with session.get("https://www.bing.com/images/search", params=params) as resp:
                    resp.raise_for_status()
                    html = await resp.text()
                    soup = BeautifulSoup(html, 'lxml')
                    elements = soup.find_all("a", class_="iusc")
                    if not elements:
                        logger.info("PicSearch Scraper: No more images found or page structure changed.")
                        break

                    new_found = 0
                    for el in elements:
                        if 'm' in el.attrs:
                            try:
                                data = json.loads(el.attrs['m'])
                                url = data.get('murl')
                                if url and url not in seen:
                                    urls.append(url)
                                    seen.add(url)
                                    new_found += 1
                                    if len(urls) >= count:
                                        break
                            except (json.JSONDecodeError, KeyError):
                                continue
                    
                    if new_found == 0 and first > 0:
                        logger.info("PicSearch Scraper: No new images found on this page, stopping.")
                        break
                    
                    first += len(elements)
                    if len(urls) >= count:
                        break
                    await asyncio.sleep(0.5)  # Be polite
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"PicSearch Scraper: Request error: {e}")
                break
            except Exception as e:
                logger.error(f"PicSearch Scraper: An unexpected error occurred: {e}", exc_info=True)
                break
    return urls[:count]