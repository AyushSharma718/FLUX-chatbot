from playwright.sync_api import sync_playwright
from urllib.parse import urljoin, urlparse
import json
import os
import time


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://www.flux.org.in/"
MAX_PAGES = 30

visited = set()
pages = []


# ============================================================
# POSSIBLE CHROME LOCATIONS
# ============================================================

CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(
        r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"
    ),
]


# ============================================================
# FIND INSTALLED CHROME
# ============================================================

def find_chrome():

    for path in CHROME_PATHS:

        if os.path.exists(path):
            print(f"Chrome found: {path}")
            return path

    return None


# ============================================================
# CHECK SAME DOMAIN
# ============================================================

def is_same_domain(url):

    base_domain = urlparse(BASE_URL).netloc
    current_domain = urlparse(url).netloc

    return current_domain == base_domain


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    lines = []

    for line in text.splitlines():

        line = " ".join(line.split())

        if line:
            lines.append(line)

    return "\n".join(lines)


# ============================================================
# CRAWL PAGE
# ============================================================

def crawl_page(page, url):

    if url in visited:
        return

    if len(visited) >= MAX_PAGES:
        return

    visited.add(url)

    print()
    print("=" * 60)
    print(f"Crawling page {len(visited)}/{MAX_PAGES}")
    print(url)
    print("=" * 60)

    try:

        # ----------------------------------------------------
        # OPEN PAGE
        # ----------------------------------------------------

        response = page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=60000
        )

        if response:
            print("HTTP status:", response.status)

        # ----------------------------------------------------
        # WAIT FOR JAVASCRIPT
        # ----------------------------------------------------

        page.wait_for_timeout(5000)

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        title = page.title()

        print("Title:", title)

        # ----------------------------------------------------
        # EXTRACT VISIBLE TEXT
        # ----------------------------------------------------

        text = page.locator("body").inner_text()

        text = clean_text(text)

        print("Text length:", len(text))

        if text:

            print()
            print("Preview:")
            print("-" * 60)
            print(text[:1000])
            print("-" * 60)

            pages.append({
                "url": url,
                "title": title,
                "content": text
            })

        else:

            print("WARNING: No visible text found.")

        # ----------------------------------------------------
        # FIND LINKS
        # ----------------------------------------------------

        links = page.locator("a").evaluate_all(
            """
            elements => elements.map(a => ({
                href: a.href,
                text: a.innerText
            }))
            """
        )

        print()
        print("Links found:", len(links))

        # ----------------------------------------------------
        # PROCESS LINKS
        # ----------------------------------------------------

        for link in links:

            next_url = link.get("href")

            if not next_url:
                continue

            # Remove fragments
            next_url = next_url.split("#")[0]

            if not next_url:
                continue

            # Only Flux website
            if not is_same_domain(next_url):
                continue

            # Already visited
            if next_url in visited:
                continue

            # Page limit
            if len(visited) >= MAX_PAGES:
                break

            crawl_page(page, next_url)

            time.sleep(1)

    except Exception as e:

        print()
        print("ERROR while crawling:")
        print(e)


# ============================================================
# SAVE KNOWLEDGE BASE
# ============================================================

def save_data():

    output_dir = os.path.join(
        os.path.dirname(__file__),
        "data"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_path = os.path.join(
        output_dir,
        "website_content.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            pages,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print("KNOWLEDGE BASE CREATED")
    print("=" * 60)

    print("Total pages:", len(pages))
    print("Output file:", output_path)

    # --------------------------------------------------------
    # SHOW TOTAL TEXT
    # --------------------------------------------------------

    total_characters = sum(
        len(page["content"])
        for page in pages
    )

    print("Total characters:", total_characters)

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("FLUX AI CHATBOT - WEBSITE CRAWLER")
    print("=" * 60)

    print()
    print("Website:", BASE_URL)
    print("Maximum pages:", MAX_PAGES)

    # --------------------------------------------------------
    # FIND CHROME
    # --------------------------------------------------------

    chrome_path = find_chrome()

    if not chrome_path:

        print()
        print("=" * 60)
        print("ERROR: GOOGLE CHROME NOT FOUND")
        print("=" * 60)

        print()
        print("Checked these locations:")

        for path in CHROME_PATHS:
            print(path)

        print()
        print("Please install Google Chrome or tell me")
        print("where Chrome is installed.")

        return

    # --------------------------------------------------------
    # START PLAYWRIGHT
    # --------------------------------------------------------

    print()
    print("Starting Chrome...")

    with sync_playwright() as playwright:

        browser = playwright.chromium.launch(
            headless=True,
            executable_path=chrome_path
        )

        page = browser.new_page(
            viewport={
                "width": 1440,
                "height": 900
            },
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/153.0.0.0 Safari/537.36"
            )
        )

        # ----------------------------------------------------
        # START CRAWLING
        # ----------------------------------------------------

        crawl_page(
            page,
            BASE_URL
        )

        # ----------------------------------------------------
        # CLOSE BROWSER
        # ----------------------------------------------------

        browser.close()

    # --------------------------------------------------------
    # SAVE DATA
    # --------------------------------------------------------

    save_data()

    print()
    print("Crawler finished successfully.")


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
