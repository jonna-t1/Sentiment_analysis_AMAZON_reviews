import pandas as pd
from colorama import Fore
from selenium import webdriver
import os

cwd = os.getcwd()
driver = webdriver.Chrome(cwd+"\chromedriver.exe")
driver.maximize_window()

driver.get("http://localhost:8080/")

driver.find_element_by_id("loginDrop").click()
driver.find_element_by_link_text("Login").click()

driver.save_screenshot("input_test_reviews/login.png")

driver.find_element_by_id("id_username").send_keys("jtur")
driver.find_element_by_id("id_password").send_keys("Sabore123")
driver.find_element_by_id("login-button").click()

driver.get("http://localhost:8080/dataView")

print("testing input")
driver.find_element_by_xpath('//input[@type="number"]').send_keys('12313451')
driver.find_element_by_xpath('//button[@type="submit"]').click()
# driver.find_element_by_id("search-button").click()
driver.save_screenshot("input_test_reviews/noTableOut.png")

## trying to enter correct keys
driver.find_element_by_xpath('//input[@type="number"]').send_keys('23')
driver.find_element_by_xpath('//button[@type="submit"]').click()
text = driver.find_element_by_class_name("first-row").text
driver.save_screenshot("input_test_reviews/inputValsShown.png")

print(text)
if text == "23":
    print(Fore.GREEN + "Test success")
else:
    print(Fore.RED + "Test failed")

driver.find_element_by_class_name("first-row").click()
driver.save_screenshot("input_test_reviews/detailView.png")

driver.find_element_by_id("correct-page").click()
match = driver.find_element_by_class_name("text-danger").click()
if match == 'Incorrect':
    print(Fore.GREEN + "Test matches")
else:
    print(Fore.RED + "Test failed")

driver.save_screenshot("input_test_reviews/incorrect.png")
driver.close()

print(Fore.WHITE+"Tested input for reviews page")

print(Fore.GREEN + "All Tests completed successfully")
