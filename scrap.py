from playwright.sync_api import sync_playwright
from utils.scrap_component import *
from utils.get_menu_data import get_menu_data
from utils.write_scrap_data_to_json import write_scrap_data_to_json
import os

NUMBER_OF_MENUS_TO_SCRAPE = 260
STARTING_URL = "https://www.bbcgoodfood.com/recipes/gluten-free-lemon-drizzle-cake"
scrapped_menus = []  # List to store scrapped menu data
scrapped_menu_urls = []  # List to track already scrapped URLs
other_menu_urls = [
    "https://www.bbcgoodfood.com/recipes/spicy-cauliflower-halloumi-rice",
    "https://www.bbcgoodfood.com/recipes/savoury-picnic-muffins",
    "https://www.bbcgoodfood.com/recipes/bicerin-coffee-chocolate-drink",
    "https://www.bbcgoodfood.com/recipes/savoury-pancake",
    "https://www.bbcgoodfood.com/recipes/satay-sweet-potato-curry",
    "https://www.bbcgoodfood.com/recipes/snack-stadium",
    "https://www.bbcgoodfood.com/recipes/halloween-treats-drinks",
    "https://www.bbcgoodfood.com/recipes/bitter-orange-poppy-seed-cake",
    "https://www.bbcgoodfood.com/recipes/spinach-muffins",
    "https://www.bbcgoodfood.com/recipes/dalgona-coffee",
    "https://www.bbcgoodfood.com/recipes/ginger-shots",
    "https://www.bbcgoodfood.com/recipes/deep-dish-meatball-marinara-pizza",
    "https://www.bbcgoodfood.com/recipes/family-meals-easy-fish-pie-recipe",
    "https://www.bbcgoodfood.com/recipes/spicy-chilli-bean-soup",
]  # List to track discovered related links

with sync_playwright() as p:
    os.system('cls')
    browser = p.chromium.launch()
    page = browser.new_page()

    while True:
        if len(scrapped_menus) == NUMBER_OF_MENUS_TO_SCRAPE:
            break

        if not scrapped_menus:
            print(f"Scraping {STARTING_URL}...")
            data = get_menu_data(page, STARTING_URL)
            scrapped_menu_urls.append(STARTING_URL)
            other_menu_urls.extend(data.related_links or [])
            scrapped_menus.append(data)
        else:
            not_indexed_urls = [url for url in other_menu_urls if url not in scrapped_menu_urls]

            if not_indexed_urls:
                print(f"Scraping {not_indexed_urls[0]}...")
                data = get_menu_data(page, not_indexed_urls[0])
                scrapped_menu_urls.append(not_indexed_urls[0])
                other_menu_urls.extend(data.related_links or [])
                scrapped_menus.append(data)
            else:
                print("No more related links left, closing...")
                break
            
    write_scrap_data_to_json(scrapped_menus)
    browser.close()