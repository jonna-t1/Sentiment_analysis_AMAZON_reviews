from selenium import webdriver
import os

cwd = os.getcwd()
driver = webdriver.Chrome(cwd+"\chromedriver.exe")
driver.maximize_window()
driver.get("http://localhost:8080")
driver.find_element_by_id("login").click()
driver.find_element_by_id("id_username").send_keys("jtur")
driver.find_element_by_id("id_password").send_keys("Sabore123")
driver.find_element_by_id("login-button").click()

driver.find_element_by_class_name("article-title").click()
#test like button
driver.find_element_by_xpath('//button[text()="Like"]').click()

#test dislike button
driver.find_element_by_xpath('//button[text()="Dislike"]').click()

#test comment section
print("Text input to the comment section: test")
driver.find_element_by_id("id_content").send_keys("test")
driver.find_element_by_xpath('//input[@value="Submit"]').click()
val = driver.find_element_by_class_name("mb-0").text
print("Text in the comment section: "+val)