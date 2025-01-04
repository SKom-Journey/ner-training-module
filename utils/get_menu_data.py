from playwright.sync_api import Page
from models.MenuData import MenuData
from utils.scrap_component import *

def get_menu_data(page: Page, source: str) -> MenuData:
    # Navigate to the source URL
    page.goto(source)

    # Wait till all the data showed
    page.wait_for_timeout(1000)

    # Extract data using previously defined functions
    title = get_menu_name(page)
    description = get_menu_description(page)
    ingredients = get_menu_ingredients(page)
    image_url = get_menu_image_url(page)
    tags = get_menu_tags(page)
    comments = get_menu_comments(page)
    nutritions = get_menu_nutritions(page)
    tips = get_menu_tips(page)
    steps = get_menu_steps(page)
    related_links = get_menu_related_links(page)

    # Return an instance of MenuData
    return MenuData(
        source=source,
        tips=tips,
        nutritions=nutritions,
        steps=steps,
        tags=tags,
        comments=comments,
        title=title,
        description=description,
        image_url=image_url,
        ingredients=ingredients,
        related_links=related_links,
    )
