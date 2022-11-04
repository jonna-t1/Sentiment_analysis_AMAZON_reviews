from selenium import webdriver
import os

driver = webdriver.Chrome("C:\\Users\j.turnbull\PycharmProjects\mysite\chromedriver.exe")
driver.maximize_window()
driver.get("http://localhost:8080/admin")


driver.find_element_by_id("id_username").send_keys("jtur")
driver.find_element_by_id("id_password").send_keys("Passy123")

driver.find_element_by_xpath('//input[@value="Log in"]').click()
driver.find_element_by_link_text("Users").click()
driver.find_element_by_class_name("addlink").click()

driver.find_element_by_id("id_username").send_keys("test")
driver.find_element_by_id("id_password1").send_keys("testing321")
driver.find_element_by_id("id_password2").send_keys("testing321")

driver.find_element_by_class_name("default").click()
driver.find_element_by_link_text("Home").click()
driver.find_element_by_link_text("test").click()
driver.find_element_by_class_name("deletelink").click()
driver.find_element_by_xpath('//input[@value="Yes, I\'m sure"]').click()