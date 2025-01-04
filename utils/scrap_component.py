from playwright.sync_api import Page
from utils.to_title_case import to_title_case
from urllib.parse import urljoin, urlparse

def get_menu_name(page: Page) -> str:
    element = page.query_selector(
        '.headline.post-header__title.post-header__title--masthead-layout h1.heading-1'
    )
    text_content = element.text_content() if element else ""
    return to_title_case(text_content.strip() if text_content else "")

def get_menu_description(page: Page) -> str:
    element = page.query_selector(
        '.editor-content.post-header__description.mt-sm.pr-xxs.hidden-print p'
    )
    return (element.text_content().strip()) if element else ""

def get_menu_nutritions(page: Page) -> list[str]:
    elements = page.query_selector_all('.nutrition-list__item')
    results = []

    for element in elements:
        if element:
            # Extract the label (e.g., "salt") from the <span>
            label = element.query_selector('span.fw-600').text_content().strip()

            # Extract the rest of the content excluding the label
            value = element.text_content().replace(label, '').strip()

            # Remove any additional text (e.g., "low") from the value
            additional_text_element = element.query_selector('.nutrition-list__additional-text')
            additional_text = additional_text_element.text_content().strip() if additional_text_element else ""

            # If there's additional text, separate it cleanly
            if additional_text:
                value = value.replace(additional_text, '').strip()
                result = f"{label} {value} {additional_text}"
            else:
                result = f"{label} {value}"

            results.append(result)

    return results

def get_menu_ingredients(page: Page) -> list[str]:
    elements = page.query_selector_all('.ingredients-list__item')
    return [element.text_content().strip() for element in elements if element]

def get_menu_tips(page: Page) -> list[str]:
    elements = page.query_selector_all('.highlight-box__content.editor-content p')
    return [element.text_content().strip() for element in elements if element]

def get_menu_steps(page: Page) -> list[str]:
    elements = page.query_selector_all('.method-steps__list-item .editor-content p')
    return [element.text_content().strip() for element in elements if element]

def get_menu_tags(page: Page) -> list[str]:
    elements = page.query_selector_all('.terms-icons-list.d-flex.post-header__term-icons-list.mt-sm.hidden-print.list.list--horizontal .terms-icons-list__text.d-flex.align-items-center')
    return [element.text_content().strip() for element in elements if element]

def get_menu_comments(page: Page) -> list[str]:
    elements = page.query_selector_all('#commentsFeed .reaction.reaction--parent .mt-reset.mb-reset .mt-reset')
    return [element.text_content().strip() for element in elements if element]

def get_menu_image_url(page: Page) -> str:
    try:
        element = page.query_selector('.post-header__image-container .image__img')
        return element.get_attribute('src') if element else ""
    except Exception:
        fallback_element = page.query_selector(
            '.image.chromatic-ignore.bg-regular.image--fluid.image--reserved-space-fallback .image__img'
        )
        return fallback_element.get_attribute('src') if fallback_element else ""

def get_menu_related_links(page: Page) -> list[str]:
    # Extract all anchor elements
    elements = page.query_selector_all('a')
    # Get the links from the elements
    links = [element.get_attribute('href') for element in elements if element]
    # The base URL (hostname) of the page
    base_url = page.url
    
    # Process links to ensure they have a hostname
    full_links = []
    for link in links:
        if link:
            # Check if the link has a hostname
            if not urlparse(link).netloc:
                # If not, append the base URL's hostname
                link = urljoin(base_url, link)
            full_links.append(link)
    
    # Filter and return only valid recipe links
    return [
        link for link in full_links 
        if '/recipes/' in link 
        and not any(x in link for x in ['/collection/', '/category/', '#'])
    ]