自動化でいちかジャンケン2020を攻略
===


### いちかジャンケンとは
これ。KONAMIのリズムゲームのイベント。[本田圭佑](https://nlab.itmedia.co.jp/nl/articles/1904/19/news144.html)より勝てる。

https://p.eagate.573.jp/game/bemani/bjm2020/


### じゃんけんするのめんどくさすぎ問題
このイベントでは一日三回、特定の時間にブラウザ上でじゃんけんすることが必要。

2020/06/29時点では毎日以下の時間帯に操作する必要がある(逃した操作は取り戻せない)。
- 1回目：10:00～15:00
- 2回目：15:00～20:00
- 3回目：20:00～10:00

いや、普通に忘れるしめんどくさいわ。

### SeleniumによるWeb操作の自動化
面倒な繰り返し操作はPythonにやらせる。
プログラムをガーッっと書き。

https://github.com/masakiaota/ichika_jannkenn

実行する。特定の時間になると勝手にwindowが開き、勝手にじゃんけんしてくれる。

![](ichika_jannkenn.gif)

こうしてほっとくだけで楽曲を解禁できるようになったのであった。


バグ報告などはgithubのissuesからお願いします。


ここからおまけちょっと技術的なお話


### おまけ：なんのライブラリを使うの？
今回求められるようなブラウザ上の操作は、`selenium`を使うことで自動化可能である。

https://www.selenium.dev/selenium/docs/api/py/

### おまけ：ログインはどうするの？
KONAMIはログイン時に画像認証が必要である。この部分まで自動化するのはちょっと難しい。

→初回ログイン時のCookieを保存しておいて、二回目起動時に読み込むことで問題を回避

参考
https://engineeeer.com/python-selenium-chrome-login/



