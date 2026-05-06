#!/usr/bin/env python3
"""
Product Image Downloader
- Barcode → Product Name (Open Food Facts) → Image
- If barcode lookup fails, use barcode as search query
- Or directly search by product name
"""

import os
import re
import time
import urllib.parse
import requests
from ddgs import DDGS

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
SEARCH_TIMEOUT = 20

# ------------------------------------------------------------
# 1. Barcode -> Product Name (Open Food Facts)
# ------------------------------------------------------------
def get_product_name_from_barcode(barcode: str) -> str:
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        if data.get("status") == 1:
            name = data["product"].get("product_name", "").strip()
            if name:
                return name
    except Exception:
        pass
    return ""

# ------------------------------------------------------------
# 2. Search image using DuckDuckGo + fallback (with retry)
# ------------------------------------------------------------
def _is_retryable_error(error: Exception) -> bool:
    err = str(error).lower()
    retry_words = (
        "timeout",
        "timed out",
        "ratelimit",
        "rate limit",
        "429",
        "403",
        "temporarily",
        "connection",
    )
    return any(word in err for word in retry_words)


def _ddgs_image_search(query: str, timeout: int = SEARCH_TIMEOUT) -> str:
    try:
        ddgs = DDGS(timeout=timeout)
    except TypeError:
        ddgs = DDGS()

    with ddgs:
        results = list(ddgs.images(query, max_results=1))
        if results:
            return results[0].get("image", "")
    return ""


def _bing_image_search(query: str, timeout: int = SEARCH_TIMEOUT) -> str:
    params = urllib.parse.urlencode({"q": query, "first": "1"})
    url = f"https://www.bing.com/images/search?{params}"
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    # Bing stores original image URLs in murl fields inside the HTML.
    matches = re.findall(r'"murl":"(https?:\\?/\\?/[^"]+)"', resp.text)
    for match in matches:
        image_url = match.replace("\\/", "/")
        if image_url.lower().startswith(("http://", "https://")):
            return image_url
    return ""


def search_image(query: str, max_retries=4) -> str:
    engines = (
        ("duckduckgo", _ddgs_image_search),
        ("bing", _bing_image_search),
    )

    for engine_name, engine in engines:
        print(f"🔎 Trying {engine_name}...")
        for attempt in range(max_retries):
            wait = min(2 ** attempt, 10)
            try:
                image_url = engine(query)
                if image_url:
                    return image_url
                print(f"⚠️ No result from {engine_name}.")
                break
            except Exception as e:
                if _is_retryable_error(e) and attempt < max_retries - 1:
                    print(f"⏳ {engine_name} error: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                    continue

                print(f"⚠️ {engine_name} failed: {e}")
                break

    return ""

# ------------------------------------------------------------
# 3. Download image
# ------------------------------------------------------------
def download_image(url: str, filename: str) -> bool:
    try:
        headers = {"User-Agent": USER_AGENT}
        r = requests.get(url, headers=headers, stream=True, timeout=15)
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        size = os.path.getsize(filename) / 1024
        print(f"✅ Saved: {filename} ({size:.1f} KB)")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

# ------------------------------------------------------------
# 4. Main (feature: product name search & image get)
# ------------------------------------------------------------
def main():
    print("🔍 Product Image Downloader")
    print("   (Enter barcode OR product name)\n")
    user_input = input("> ").strip()
    if not user_input:
        return

    # --- Step 1: Determine search term ---
    search_term = ""
    # If input looks like a barcode (8-14 digits)
    if user_input.isdigit() and 8 <= len(user_input) <= 14:
        print("📦 Barcode detected. Looking up product name...")
        name = get_product_name_from_barcode(user_input)
        if name:
            print(f"✅ Found: {name}")
            search_term = name
        else:
            print(f"⚠️ Barcode '{user_input}' not found in Open Food Facts.")
            print("   Falling back to use barcode as search query.")
            search_term = user_input
    else:
        # Input is already a product name
        search_term = user_input
        print(f"📝 Using product name: {search_term}")

    # --- Step 2: Search for image ---
    query = f"{search_term} product"
    print(f"\n🔎 Searching images for: {query}")
    img_url = search_image(query)
    if not img_url:
        print("❌ No image found after retries.")
        return

    # --- Step 3: Download ---
    safe_filename = "".join(c if c.isalnum() else "_" for c in search_term)[:50] + ".jpg"
    print(f"💾 Downloading from: {img_url[:80]}...")
    download_image(img_url, safe_filename)

if __name__ == "__main__":
    main()
