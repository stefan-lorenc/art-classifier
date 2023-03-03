from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options


def driver_creation(is_headless=False, is_eager=False):
    service = ChromeService(executable_path=ChromeDriverManager().install())
    chrome_options = Options()
    if is_headless == True:
        chrome_options.add_argument("--headless")
    if is_eager == True:
        chrome_options.page_load_strategy = 'eager'
    # noinspection PyArgumentList
    driver = webdriver.Chrome(options=chrome_options, service=service)
    return driver

