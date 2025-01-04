from playwright.sync_api import Page
from models.SymptomAllergyData import SymptomAllergyData
from utils.scrap_symptom_allergy_component import *

def get_symptom_allergy_data(page: Page, source: str) -> SymptomAllergyData:
    # Navigate to the source URL
    page.goto(source)

    # Wait till all the data showed
    page.wait_for_timeout(1000)

    # Extract data using previously defined functions
    title = get_symptom_title(page)
    paragraphs = get_symptom_paragraphs(page)
    headings = get_symptom_headings(page)

    # Return an instance of SymptomAllergyData
    return SymptomAllergyData(
        source=source,
        title=title,
        headings=headings,
        paragraphs=paragraphs,
    )
