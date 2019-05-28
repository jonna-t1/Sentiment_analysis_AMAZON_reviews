import datetime
import time

from colorama import Fore
from selenium import webdriver
import os

# print(datetime.date.today().strftime('%b %d, %Y CONTACT'))
# exit("sterp")

cwd = os.getcwd()
driver = webdriver.Chrome(cwd+"\chromedriver.exe")
driver.maximize_window()

driver.get("http://localhost:8080")
driver.find_element_by_id("loginDrop").click()
driver.find_element_by_link_text("Login").click()

driver.save_screenshot("RequestCRUD/login.png")

driver.find_element_by_id("id_username").send_keys("jtur")
driver.find_element_by_id("id_password").send_keys("Sabore123")
driver.find_element_by_id("login-button").click()

driver.save_screenshot("RequestCRUD/login2.png")

print("end of login test")

# add request
email = 'jtur@accenture.com'

driver.get("http://localhost:8080/admin/tracker/request/")
driver.save_screenshot("RequestCRUD/preRequest.png")
driver.get("http://localhost:8080")

driver.save_screenshot("RequestCRUD/home.png")

driver.find_element_by_id("loginDrop").click()
time.sleep(2)
driver.find_element_by_link_text("Make a request").click()
driver.find_element_by_id("id_name").send_keys("jon")
driver.find_element_by_id("id_surname").send_keys("Tur")
driver.find_element_by_id("id_email").send_keys(email)
driver.find_element_by_id("id_contactNo").send_keys("077834203239")
# driver.find_element_by_id("id_reason").send_keys("Tur")
driver.find_element_by_xpath("//select[@name='reason']/option[text()='CONTACT']").click()
driver.find_element_by_id("id_other").send_keys("I would like to contact you about...")
driver.save_screenshot("RequestCRUD/completeForm.png")

driver.find_element_by_xpath('//button[text()="Post"]').click()
driver.save_screenshot("RequestCRUD/confirmation.png")

driver.find_element_by_link_text("Dashboard").click()
driver.save_screenshot("RequestCRUD/displayCount.png")
driver.find_element_by_id("messagesDropdown").click()
driver.save_screenshot("RequestCRUD/displayDrop.png")
driver.find_element_by_link_text("Read More Messages").click()
driver.save_screenshot("RequestCRUD/admin.png")

# print(datetime.date.today())
date = datetime.date.today().strftime('%b %d, %Y - CONTACT')
driver.find_element_by_link_text(date).click()
input  = driver.find_element_by_xpath('//input[@type="email"]')
contents = input.get_attribute('value')

if contents == email:
    print("Test passed, content there and matches")
else:
    print("Test failed")

# #delete
driver.find_element_by_link_text("Delete").click()
driver.save_screenshot("RequestCRUD/preDelete.png")
driver.find_element_by_xpath('//input[@type="submit"]').click()
driver.save_screenshot("RequestCRUD/postDelete.png")
driver.close()
print(Fore.GREEN + "All Tests completed successfully")
