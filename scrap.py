from playwright.sync_api import sync_playwright
from utils.scrap_component import *
from utils.get_menu_data import get_menu_data
from utils.write_scrap_data_to_json import write_scrap_data_to_json
import os
from utils.get_symptom_allergy_data import get_symptom_allergy_data

NUMBER_OF_MENUS_TO_SCRAPE = 260
STARTING_URL = "https://www.bbcgoodfood.com/recipes/gluten-free-lemon-drizzle-cake"
scrapped_menus = []  # List to store scrapped menu data
scrapped_symptoms = []  # List to store scrapped menu data
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
symptom_allergy_urls = [
    # Sneeze
    "https://www.webmd.com/pets/cats/why-cats-sneeze",
    "https://www.webmd.com/men/features/medical-mysteries-she-wrote",
    "https://www.webmd.com/pets/dogs/what-is-reverse-sneezing-dogs",

    # Cough
    "https://www.webmd.com/cold-and-flu/overview",
    "https://www.webmd.com/pets/dogs/kennel-cough-in-dogs",
    "https://www.webmd.com/children/remedies-toddler-cough",
    "https://www.webmd.com/asthma/cough-variant-asthma",

    # Itch
    "https://www.webmd.com/skin-problems-and-treatments/eczema/features/cm/how-to-manage-unbearable-itch",
    "https://www.webmd.com/skin-problems-and-treatments/why-so-itchy",
    "https://www.webmd.com/skin-problems-and-treatments/eczema/features/cm/managing-itch-from-eczema",

    # Vomit
    "https://www.webmd.com/digestive-disorders/what-to-know-throwing-up-foam",
    "https://www.webmd.com/anxiety-panic/what-is-emetophobia",

    # Stomachache
    "https://www.webmd.com/first-aid/stomachache-and-nausea-children",
    "https://www.webmd.com/parenting/features/anxiety-stress-and-stomachaches",
    "https://www.webmd.com/parenting/baby/help-child-tummyache",

    # Swallow
    "https://www.webmd.com/digestive-disorders/swallowing-problems",
    "https://www.webmd.com/parkinsons-disease/parkinsons-disease-swallowing-problems",
    "https://www.webmd.com/a-to-z-guides/what-to-know-about-swallowing-pills",

    # Abdominal Pain
    "https://www.webmd.com/baby/abdominal-pain-and-pregnancy-what-to-know",
    "https://www.webmd.com/a-to-z-guides/what-to-know-abdominal-mass",
    "https://www.webmd.com/first-aid/abdominal-pain-in-children-treatment",
    
    # Diarrhea
    "https://www.webmd.com/digestive-disorders/diarrhea-stomach-flu",
    "https://www.webmd.com/parenting/baby/baby-diarrhea-causes-treatment",

    # Rash
    "https://www.webmd.com/skin-problems-and-treatments/understanding-heat-rash-basics",
    "https://www.webmd.com/skin-problems-and-treatments/eczema/atopic-vs-contact-dermatitis",

    # Nausea
    "https://www.webmd.com/digestive-disorders/remedies-for-nausea-upset-stomach",
    "https://www.webmd.com/migraines-headaches/headaches-migraines-and-nausea",
    
] # List of symptom allergies URLs (WebMD)

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

    # Scrap symptom allergies
    for url in symptom_allergy_urls:
        print(f"Scraping {url}...")
        data = get_symptom_allergy_data(page, url)
        scrapped_symptoms.append(data)

    write_scrap_data_to_json(scrapped_symptoms, 'symptoms_')
    write_scrap_data_to_json(scrapped_menus, 'menus_')
    browser.close()