scikit-learnを改変しよう(良いタイトルを考えよう)
===

### 書きたい内容
- 本記事をおすすめしない人
- scikit-learnのディレクトリ構造の俯瞰
- 環境のおすすめ
- 編集した内容を即時反映する。pip --editable
- 編集に必要な知識(リンク集にしよう
  - (pythonの知識?)
  - scikit-learnの自作クラスを作るには？のリンクを貼る
  - Cythonの知識
- 実際にやってみよう
  - dtreevisをDecisionTreeClassifierに組み込んでみる。

### はじめに
この記事を読むことで、scikit-learnの中身のコードに改変を加えることができるようになることを期待している。改変に必要な知識も学習できるようリンクを用意してある。そして改変を加えたコードをpipで管理する方法も示した。
最後にはdtreevisという決定木の可視化ライブラリをDecisionTreeClassifierのメソッドとして組み込む具体例を示す。

#### 本記事をおすすめしない人
- Pythonのリスト、タプル、辞書の違いは？と聞かれて答えられない方。
  - 前半(6章まで)だけでいいのでPythonのチュートリアルをやりましょう。
  - https://docs.python.org/ja/3/tutorial/
- 自分のPythonの環境をすぐに切り替えられない方
  - まずは仮想環境の作り方やpyenvの使い方を覚えましょう
- 属性(attribute,メンバ)、メソッド、継承と言われてさっぱりな方
  - Pythonのクラスについて勉強しましょう
  - https://docs.python.org/ja/3/tutorial/classes.html

Pythonの基礎がわかっていればこの記事を読むのに苦労することは全くない。ただ、機械学習アルゴリズムに精通している方でもおすすめしない人に当てはまる方は苦労するかもしれない。

本記事をより良く理解するためには、実際に仮想環境(もしくはpyenvで環境ごと切り替え)を作り、実際にやってみることである。

既存のアルゴリズムに改変を加える研究をするぜ！と意気込む大学生の手助けになることを祈っている。

(オライリー本的な導入になってしまった。)

### scikit-learnのディレクトリ構造の俯瞰
まずは、scikit-learnのディレクトリ構造を俯瞰してみよう。

githubのページはこちらだ。
https://github.com/scikit-learn/scikit-learn

`./`には、`README.md`をはじめとする書類系のファイルと`setup.py`をはじめとするinstallに必要なファイルがある。ディレクトリの方を見てみると`doc/`や`eaxmples/`などユーザーにとってありがたいファイルが入っているであろうディレクトリがある。

学習器が実際に記述されているのは`sklearn/`の中だ。こちらの中を見ていこう。クリックすると[こちら](https://github.com/scikit-learn/scikit-learn/tree/master/sklearn
)のページに飛ぶはずだ。

大枠としてのアルゴリズムの名称、もしくは処理の名称がディレクトリの名称としてつけられている。この一覧の中から自分が手を加えたいものをファイルを探して改変するわけである。

たとえば[Treeの中](https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/tree)を見てみよう。

中身はこういう構造になっている。
```
├── __init__.py
├── _criterion.pxd
├── _criterion.pyx
├── _reingold_tilford.py
├── _splitter.pxd
├── _splitter.pyx
├── _tree.pxd
├── _tree.pyx
├── _utils.pxd
├── _utils.pyx
├── export.py
├── setup.py
├── tests
│   ├── __init__.py
│   ├── test_export.py
│   ├── test_reingold_tilford.py
│   └── test_tree.py
└── tree.py
```

すぐに`_`から始まっているファイル名があるのに気がつくだろう。scikit-learnではユーザーが直接使わないファイルは`_`から始まっている。この法則は一部のファイルに記載されている関数にも適応される。
(例えば[forest.py](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/forest.py)の冒頭で定義されている関数を見ればわかるだろう)

また、.py以外の拡張子も存在することに気がつくだろう。`.pyx`はCythonのファイルである。また`.pxd`はCythonの補助ファイルである。Cythonについては後述する。

編集を始める前に、環境を整えて置こう。また編集したscikit-learnをimportしたときに編集が反映されるようにしておこう。

### 開発環境を整える


### 