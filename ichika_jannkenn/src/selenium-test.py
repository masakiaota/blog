# coding:utf-8
# from: https://qiita.com/y__ueda/items/7b6f2a95ea45667e1029
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from time import sleep

# ブラウザを開く。
driver = webdriver.Chrome()
# Googleの検索TOP画面を開く。
driver.get("https://www.google.co.jp/")
sleep(2)
# 検索語として「selenium」と入力し、Enterキーを押す。
driver.find_element_by_name("q").send_keys("selenium")
driver.find_element_by_name("q").send_keys(Keys.ENTER)
# タイトルに「Selenium - Web Browser Automation」と一致するリンクをクリックする。
sleep(2)
driver.find_element_by_link_text("Seleniumn").click() #ここでエラーが出るが
# 5秒間待機してみる。
sleep(5)
# ブラウザを終了する。
driver.close()
