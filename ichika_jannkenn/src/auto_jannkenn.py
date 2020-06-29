from selenium import webdriver  # brew cask でchromedriverのinstallが必要
from selenium.webdriver.common.keys import Keys
import time
from time import sleep
import datetime
import schedule
from pathlib import Path
from random import choice

hands = [0, 1, 2]  # グーチョキパー


def do_jannkenn():
    print('じゃんけんed at', datetime.datetime.now())

    # 初回ログイン時にクッキー情報を取得
    userdata_dir = Path('./userdata/')
    if not userdata_dir.exists():
        userdata_dir.mkdir()

    option = webdriver.ChromeOptions()
    option.add_argument('--user-data-dir=' +
                        userdata_dir.as_posix())  # 完全に閉じないと次回error

    # じゃんけん画面へ
    driver = webdriver.Chrome(options=option)
    driver.get("https://p.eagate.573.jp/game/bemani/bjm2020/janken/index.html")
    sleep(1)

    # じゃんけん済みだったらなにもせず次の回まで待機
    element_janken = driver.find_elements_by_id('janken')
    if element_janken[0].get_attribute('innerHTML').count('じゃんけん済み'):
        pass  # なにもしない処理をする
    else:
        # じゃんけんする
        hand = choice(hands)
        element_janken[hand].click()  # TODO 要素みて正しいかverify
        # driver.find_element_by_link_text("hoge").click()

    # ブラウザを終了
    # driver.close()
    driver.quit()


schedule.every().day.at('10:05').do(do_jannkenn)
schedule.every().day.at('15:05').do(do_jannkenn)
schedule.every().day.at('20:05').do(do_jannkenn)


while True:
    schedule.run_pending()
    time.sleep(60 * 10)  # 10分おきに実行可能かcheck


# test #ok!
# schedule.every().day.at('22:32').do(do_jannkenn)
# schedule.every().day.at('22:33').do(do_jannkenn)
# schedule.every().day.at('22:34').do(do_jannkenn)
