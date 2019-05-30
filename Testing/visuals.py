import datetime
import time

from colorama import Fore
from selenium import webdriver
import os


cwd = os.getcwd()
driver = webdriver.Chrome(cwd+"\chromedriver.exe")
driver.maximize_window()

driver.get("http://localhost:8080")
driver.find_element_by_id("loginDrop").click()
driver.find_element_by_link_text("Login").click()
driver.find_element_by_id("id_username").send_keys("jtur")
driver.find_element_by_id("id_password").send_keys("Sabore123")
driver.find_element_by_id("login-button").click()

driver.get("http://localhost:8080")
driver.find_element_by_id("visuals").click()
time.sleep(2)
driver.find_element_by_id("line").click()
driver.save_screenshot("visuals/LineChart.png")

driver.get("http://localhost:8080")
driver.find_element_by_id("visuals").click()
time.sleep(2)
driver.find_element_by_id("pie").click()
driver.save_screenshot("visuals/PieChart.png")

driver.close()
print(Fore.GREEN + "All Tests completed successfully")
