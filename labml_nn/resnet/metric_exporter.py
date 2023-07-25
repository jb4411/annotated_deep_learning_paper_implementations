import os
import time

import requests
from selenium.webdriver.common.keys import Keys
import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions
from bs4 import BeautifulSoup
import bs4

def main():
    base_url = "http://localhost:5005/run/"
    if len(sys.argv) > 1:
        url = base_url + sys.argv[1]
    else:
        inpt = input("Please enter the run ID: ")
        if inpt.strip() == "":
            return
        url = base_url + inpt

    while True:
        response = requests.get(url)
        if response.status_code == 404:
            print("Error: 404 Page not found!", file=sys.stderr)
            inpt = input("Please enter a valid run ID: ")
            if inpt.strip() == "":
                return
            url = base_url + inpt
        else:
            break

    config_page = requests.get(url + "/configs")
    soup = BeautifulSoup(config_page.text, 'html.parser')

    return
    path = os.path.join(os.getcwd(), "geckodriver-v0.33.0-win64", "geckodriver.exe")
    print(path)
    driver = webdriver.Firefox(executable_path=path, timeout=1000000)
    driver.get("http://localhost:5005/runs")
    print()


if __name__ == '__main__':
    main()
