# -*- coding: utf-8 -*-
"""
Image scraping functionality for the PicSearch plugin.
Uses a synchronous requests-based implementation running in an executor
to ensure reliable pagination cookie handling.
"""
import asyncio
import json
import time
from typing import List
import random

#本实现为确保浏览器行为一致，使用requests库进行请求，已包装在run_in_executor()内，不会导致阻塞
import requests
from bs4 import BeautifulSoup

from astrbot.api import logger

def _scrape_images_sync(query: str, count: int) -> List[str]:
    """
    Synchronously scrapes Bing Images for a given query, handling pagination.
    This function is designed to be run in a thread pool executor.
    """
    search_url = "https://www.bing.com/images/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    }

    image_urls = []
    seen_urls = set()
    
    with requests.Session() as session:
        session.headers.update(headers)
        
        first = 0
        while len(image_urls) < count:
            params = {"q": query, "first": first}
            
            try:
                logger.info(f"PicSearch Scraper: Fetching images from index {first} using requests...")
                response = session.get(search_url, params=params, timeout=20)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'lxml')
                image_elements = soup.find_all("a", class_="iusc")

                if not image_elements:
                    logger.info("PicSearch Scraper: No more images found, or page structure has changed.")
                    break

                new_images_found_this_page = 0
                for element in image_elements:
                    if 'm' in element.attrs:
                        m_attr = element.attrs['m']
                        try:
                            data = json.loads(m_attr)
                            if 'murl' in data:
                                url = data['murl']
                                if url not in seen_urls:
                                    image_urls.append(url)
                                    seen_urls.add(url)
                                    new_images_found_this_page += 1
                                    if len(image_urls) >= count:
                                        break
                        except json.JSONDecodeError:
                            continue
                
                if new_images_found_this_page == 0 and first > 0:
                    logger.info("PicSearch Scraper: No new images found on this page. Stopping.")
                    break

                first += len(image_elements)
                
                if len(image_urls) >= count:
                    break
                
                # Be polite and wait a bit before the next request
                time.sleep(random.uniform(0.5, 1.5))

            except requests.exceptions.RequestException as e:
                logger.error(f"PicSearch Scraper: An error occurred during the web request: {e}")
                break
            except Exception as e:
                logger.error(f"PicSearch Scraper: An unexpected error occurred: {e}", exc_info=True)
                break

    return image_urls[:count]

async def scrape_image_urls(query: str, count: int) -> List[str]:
    """
    Asynchronous wrapper for the synchronous scraper.
    Runs the blocking I/O in a separate thread to avoid blocking the event loop.
    """
    loop = asyncio.get_running_loop()
    logger.info("PicSearch Scraper: Switching to synchronous requests in executor for reliability.")
    # Use run_in_executor to run the synchronous function in a thread pool
    result = await loop.run_in_executor(
        None,  # Use the default executor (ThreadPoolExecutor)
        _scrape_images_sync,
        query,
        count
    )
    logger.info(f"PicSearch Scraper: Finished scraping via executor. Total URLs found: {len(result)}")
    return result