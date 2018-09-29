# Hyper+xonshで超モダンな環境づくり

### はじめに
ばんくしさんをフォローしたらすっかりxonshに洗脳された。
また、巷でうわさのHyperなるターミナルアプリを試してみたら使いやすかったので、組み合わせて使ったら最強では？と安直な発想で記事を書きはじめた。

適当に操作してみたのがこちら。

{GIF挿入}

候補がフローティングウィンドウに出ていたり、補完がゴリゴリに効いていることがわかるだろう。また、動画内ではpythonの仮想環境を切り替えてもいる。

本記事では以下のことを書く
- Hyper、xonshの紹介
- Hyperの導入
    - 起動するshellの選択
    - 文字化け対策
- xonshの導入
    - お手軽に使ってみる
    - xonshを直接起動するための設定
    - xonsh内でpythonの仮想環境を切り替える


記事の対象者
- .bashrc、.zshrcなどと言われて何かわかる方
- Homebrewがある


#### Hyperとは
デザインがイケてるターミナル。テキストファイルで設定ができるので管理がしやすそう。拡張機能の導入も簡単。(これでiterm2のごちゃごちゃした設定とはおさらば)
https://hyper.is

すでにいろいろ記事がある。こちらがよくまとめられているなと思う。

https://qiita.com/vimyum/items/44478a51ef3a6f49804f

#### xonshとは
超便利なshell。fishの使いやすさを増した感じ(Pythonが扱えるおかげで)。

https://xon.sh/sidebar.html


詳しくはこの記事を見ればどうにかなる。

https://vaaaaaanquish.hatenablog.com/entry/2018/06/22/194227

### Hyperの導入
https://hyper.is に従ってやるだけ。おわり。ここからは一旦xonshから離れる。
Hyperの設定ファイルは、~/.hyper.jsである。

#### 起動するshellの選択
デフォルトではログインshellが起動するようになっているが、Hyperの方で、どのshellを立ち上げるか決めることができる。

.hyper.js
```
...
shell: '/usr/local/bin/fish',
...
```
としてやれば、ログインシェルがbashでもhyperを立ち上げたときはfishでログインする。

xonshをメインにするならここはあとでxonshに変更する。

#### 文字化けの問題
Hyperでは日本語が文字化けする場合がある。原因はUTF-8。なので、この設定もしないと行けない。

.hyper.js
```
...
env: {LANG: 'ja_JP.UTF-8'},
...
```

また使っているshellの方も設定しないといけない。xonshでは設定しなくても文字化けすることはなかったが、bash等で文字化けが起きてしまった。なので、bashを例にすると、以下の内容をファイルの最後に追加した。

.bash_profile
```
...
export LANG=ja_JP.UTF-8
```

bash等からxonshを起動して使おうと思っている方も、これをしないといけない。文字化けするほかのshellを経由するとxonshも文字化けする。

### xonshの導入
さまざまな方法がある。軽く使ってみたい方はいつものpythonの環境で
```
pip install xonsh
xonsh
```
でかまわない。しかし、他のshellを経由に普段遣いをしたい方は、少し工夫が必要。

#### お手軽に使ってみる
上記の通り、いつもの環境で
```
pip install xonsh
xonsh
```
とすれば、起動する。

設定ファイルは.xonshrcである。どんな設定をすればいいかはだいたいばんくしさんのブログを見ればわかる。特に便利なのを示すと、以下のあたり。

.xonshrc
```
# 補完をEnterで直接実行しない
$COMPLETIONS_CONFIRM = True
# 補完時に大小区別しない
$CASE_SENSITIVE_COMPLETIONS = False
# ディレクトリ名を入力でcd
$AUTO_CD = True
# キー入力即評価（サイコー）
$UPDATE_COMPLETIONS_ON_KEYPRESS = True
```

また、プロンプトの見た目は自分は以下の設定にしている。

```
# プロンプトの表記
$PROMPT = "\n{INTENSE_GREEN}[ {cwd} ] \n{env_name:{} }{user}{WHITE}@{INTENSE_BLUE    }{hostname}{WHITE}{branch_color}{curr_branch: {}}\n{NO_COLOR}{BOLD_BLUE}{prompt_e    nd}{NO_COLOR} "
```

pythonさえあればこのシェルが導入できるので便利である。

また、xonshを自動的に起動したければ、いつも使っているshellの設定ファイルにxonshと書き込めばいい。bashならばこんな感じ。

.bashrc
```
...#一番最後に
xonsh
```

#### メインに使うには
他のshellを経由せずに使うには一工夫いる。とくにpyenvと相性は最悪である。そのため、pyenvは手放さなければいけない。しかし、voxという仮想環境を構築するコマンドをxonshはサポートしているため、困るようなことは無いだろう。

##### 導入
前述した通りお使いのshellのpyenvへのPATHを一度切っていただきたい。

bashならば、該当の部分をコメントアウトすればよい。

.bash_profile
```
...
#export PATH="$PYENV_ROOT/bin:$PATH"
#export PYENV_ROOT="$HOME/.pyenv"
#eval "$(pyenv init -)"
...
```

Mac前提の導入になるが以下のコマンドで導入する。

```
#もしHomebrewを導入していないのなら先に導入してください
brew install python3 #すでにあるならok
pip3 install xonsh
pip3 install gnureadline
pip3 install prompt-toolkit
```

これでxonshと打つと起動できるはずだ。普通に使えるように見えるが、ログインシェルをいきなりxonshに変えたり、Hyperでダイレクトにxonshが立ち上がるようにしては行けない。まだ環境変数の設定をしていないからだ。

##### 環境変数
設定ファイルに必要なPATHを追加する。人によって異なるのでなんとも言えないが、自分の場合はこう

.xonshrc
```
...
$PATH=["bin","/usr/local/bin","/usr/bin","/bin","usr/sbin", "/sbin",""]
...
```

環境変数を追加したら、Hyperの設定でxonshが直接立ち上がるように設定しよう。


.hyper.js
```
...
shell: '/usr/local/bin/xonsh',
...
```

これでHyper+xonshの超モダン環境のできあがりである。

#### Pythonの仮想環境の構築
前述した通り、xonshはpyenvと相性最悪である。では仮想環境はどうするか。公式がそのやり方を用意してくれている。

https://xon.sh/python_virtual_environments.html

まずは、.xonshrcに以下の内容を追加してほしい

```
# 仮想環境
xontrib load vox
```

そして、仮想環境を構築するには以下のコマンドを実行すれば良い。
```
vox new {好きな名前}
```

仮想環境を制作できたら、起動しよう。
```
vox activate {さっきの好きな名前}
```

これでおしまいだ。
いちいち仮想環境の起動が面倒なら、これを.xonshrcに書き込んでしまえば良い。

念の為
```
pip --version
```
でパスを確認すると、仮想環境にライブラリをインストールするようになっていることを確認できる。



