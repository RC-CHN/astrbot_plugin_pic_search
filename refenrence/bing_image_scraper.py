# -*- coding: utf-8 -*-
"""
This script provides a function to scrape a large number of image URLs from the
Bing Images search results page by handling pagination.

Required libraries:
- requests: To send HTTP requests.
- beautifulsoup4: To parse HTML content.
- lxml: A fast and efficient XML and HTML parser library used by BeautifulSoup.

You can install them using pip:
pip install requests beautifulsoup4 lxml
"""

import requests
import json
import time
from bs4 import BeautifulSoup
from typing import List

def scrape_images(query: str, count: int = 50) -> List[str]:
    """
    Scrapes Bing Images for a given query, handling pagination to fetch a
    large number of image URLs.

    Args:
        query (str): The search term for the images.
        count (int, optional): The desired number of images. The script will
                               attempt to fetch up to this number. Defaults to 50.

    Returns:
        List[str]: A list of direct URLs to the images. Returns an empty list
                   if the request fails or no images are found.
    """
    search_url = "https://www.bing.com/images/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    }

    image_urls = []
    seen_urls = set()
    
    # Use a session to persist cookies
    with requests.Session() as session:
        session.headers.update(headers)
        
        # The 'first' parameter controls pagination. It's the index of the first result.
        first = 0
        while len(image_urls) < count:
            params = {"q": query, "first": first}
            
            try:
                print(f"Fetching images from index {first}...")
                response = session.get(search_url, params=params, timeout=15)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'lxml')
                image_elements = soup.find_all("a", class_="iusc")

                if not image_elements:
                    print("No more images found, or page structure has changed.")
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
                    # If we are not on the first page and find no new images, we are done.
                    print("No new images found on this page. Stopping.")
                    break

                # Update the 'first' parameter for the next page
                first += len(image_elements)
                
                if len(image_urls) >= count:
                    break
                
                # Be polite and wait a bit before the next request
                time.sleep(1)

            except requests.exceptions.RequestException as e:
                print(f"An error occurred during the web request: {e}")
                break
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break

    return image_urls[:count]

if __name__ == '__main__':
    search_query = "tokyo street photography"
    target_count = 75
    print(f"Scraping up to {target_count} images for: '{search_query}'")
    
    urls = scrape_images(search_query, count=target_count)

    if urls:
        print(f"\nSuccessfully found {len(urls)} image URLs:")
        for i, url in enumerate(urls, 1):
            print(f"{i}. {url}")
    else:
        print("\nCould not find any images or an error occurred.")