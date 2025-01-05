from playwright.sync_api import Page
from utils.to_title_case import to_title_case

def get_symptom_title(page: Page) -> str:
    element = page.query_selector(
        '.inner-article-container h1'
    )
    text_content = element.text_content() if element else ""
    return to_title_case(text_content.strip() if text_content else "")

def get_symptom_paragraphs(page: Page) -> str:
    elements = page.query_selector_all('.inner-article-container p')
    return [element.text_content().strip() for element in elements if element]

def get_symptom_headings(page: Page) -> str:
    elements = page.query_selector_all('.inner-article-container .jumplink-headers')
    return [element.text_content().strip() for element in elements if element]

def get_symptom_bullets(page: Page) -> str:
    elements = page.query_selector_all('.inner-article-container li')
    return [element.text_content().strip() for element in elements if element]