from playwright.sync_api import Page

def screenshot(page: Page, file_name: str = "screenshot"):
    page.screenshot(path="screenshots/" + file_name + ".jpeg")
    print(f"Screenshot saved to {file_name}")