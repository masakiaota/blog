【具体例つき】scikit-learnを改変しよう ~改変版のinstall方法と改変に必要な知識のリンク集~
===

![](fig/eye_catch.png)

### はじめに
この記事を読むことで、scikit-learnの中身のコードに改変を加えることができるようになることを期待している。改変に必要な知識も学習できるようリンクを用意してある。そして改変を加えたコードをpipで管理する方法も示した。
最後には具体例として、決定木のfeature_importances_を可視化するメソッドをDecisionTreeClassifierに組み込む。

[TOC]

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
(例えば[forest.py](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/forest.py)の冒頭で定義されている関数を見ればわかるだろう。この関数はユーザーが直接使うことを想定していない。)

また、.py以外の拡張子も存在することに気がつくだろう。`.pyx`はCythonのファイルである。また`.pxd`はCythonの補助ファイルである。Cythonについては後述する。

編集を始める前に、環境を整えて置こう。また編集したscikit-learnをimportしたときに編集が反映されるようにしておこう。

### 開発環境を整える
今の自分の環境が壊れないように、別の環境で作業をしよう。ここではpythonや環境の切り替えツールの導入方法は説明しない。各自使いこなせるものを持っていることを前提とする。

新しい環境にするとパッケージ(ライブラリ)が全然入っていないと思うが、インストールは読者に任せる(とはいえ、scikit-learnだけはインストールしないでほしい)

#### pyenvを用いた方法
筆者はpyenvを使用している。pyenvを用いているならば、仮想環境を用意するよりバージョンを切り替えたほうが手っ取り早いだろう。
```
pyenv install 3.6.2 #バージョンは3系の新しめなら何でも良い
pyenv global 3.6.2 #さっきインストールしたバージョンに切り替え
```

筆者はAnaconda信者だが、あとでpipを用いることになるので`pyenv install`でAnacondaを用いることはおすすめしない。

#### venvを用いた方法
Python3ではvenvで仮想環境を構築することができる。pyenvを用いていないかたはこちらの方法で作成可能だろう。この方法でなくても慣れている方法で全然構わない。
```
python3 -m venv edit_sklearn #最後は好きな名称にしてください
source edit_sklearn/bin/activate #ここも適切なものに変えてください
```
で切り替え可能。`deactivate`で終了。


### 編集した内容が反映されるようにインストールする
さぁ、いよいよscikit-learnをインストールしよう。

#### 環境の確認
まずは今の環境にscikit-learnが入っていないことを確認したい。
```
pip list | grep scikit
```
と打って scikit-learnと表示されなければ問題ない。すでにインストールされていた場合は、一旦アンインストールしよう。
```
pip uninstall scikit-learn
```

そして、いま、作業ディレクトリにいることを確認しよう。

#### pip install --editable
ここからは公式の手順で導入することにする。

https://scikit-learn.org/stable/developers/advanced_installation.html

まずはソースコードのダウンロードからだ。
```
#すこし時間がかかります。
git clone git://github.com/scikit-learn/scikit-learn.git
```

次にscikit-learnに必要な依存パッケージをインストールする。
```
pip install numpy scipy cython pytest pandas
```

そして、いよいよinstallする。
```
cd scikit-learn #さっきgit cloneしたディレクトリに移動 (手動でダウンロードすると名前がscikit-learn-masterになります)
pip install --editable . #インストール！！
#時間がかかります
```

install時にはcythonによるC, Cppへのコンパイル、さらにそれらから実行ファイルへのコンパイル等があるので時間がかかります。気長に待ちましょう。

#### トラブルシューティング
Cをコンパイルする工程でエラーが発生しinstallが終了する可能性がある。自分の計算機にgccがインストールされていてパスが問題ないか確認しよう。

Macを使っている人だと、コンパイラの関係でエラーが出てインストールに苦労するかもしれない。これはApple Clangが`--openmp`という引数をサポートしていないのが原因だ。

```
brew install gcc #gccを導入
```
gccを導入して、コンパイルにはこれを使うように指定する。

例えばbashをお使いならば、.bashrcに以下のようなエイリアスを書き込む。
```
# .bashrcに追記する
export CC=gcc-8 #自分のgccのバージョンに合わせてください
export CXX=gcc-8
export ARCHFLAGS="-arch x86_64"
alias gcc=gcc-8
alias g++=g++-8
```

これでもだめなら、googleで検索するかstack overflowで質問をしよう。


#### 準備完了
ここまで来たら、もう準備は完了である。'git clone'したscikit-learnをいじれば、いじった内容が即時で反映される。(jupyterを起動しているときはカーネルを一回シャットダウンする必要がある。)

Cythonファイルを編集した場合には、コンパイルの必要がある。再び`scikit-learn/`の中で
```
pip install --editable .
```
を行えば良い。

もしくは、`scikit-learn/`の中で
```
python setup.py build_ext --inplace
```
としてもokだ。筆者は後者派である。

好きなようにscikit-learnを改変しよう。研究に使うもよし、バグを修正してOSSに貢献するもよしだ。

### 編集に必要な知識
編集に必要な知識を身に着けるときに役立つサイトをまとめた。辞書代わりにどうぞ。

#### Pythonの知識
がんばろう

公式日本語ドキュメント
https://docs.python.org/ja/3.7/

#### scikit-learn
引数の意味とかはソースコードよりもドキュメントのほうが早く探せる。

https://scikit-learn.org/stable/index.html

#### scikit-learn準拠モデル
自分で書いたモデルをscikit-learnとともに使うためにいくつかルールがある。scikit-learnに実装されているモデルももちろんそのルールにしたがっているので知っておくと良いだろう。なぜそれが継承されてる？という疑問も少なくなる。

https://qiita.com/roronya/items/fdf35d4f69ea62e1dd91

https://qiita.com/_takoika/items/89a7e42dd0dc964d0e29

#### Cythonの知識
実行時、速度のボトルネックになる部分はCythonを用いて実装されている。改変したい部分がCythonにある場合、知識が必要だろう。


https://cython.readthedocs.io/en/latest/

公式ドキュメントの翻訳(アクセンステクノロジの増田さんすごい！)

http://omake.accense.com/static/doc-ja/cython/index.html

Cython本 (Cythonの闇に飲み込まれたい方はおすすめです)

<div class="amazlet-box" style="margin-bottom:0px;"><div class="amazlet-image" style="float:left;margin:0px 12px 1px 0px;"><a href="http://www.amazon.co.jp/exec/obidos/ASIN/4873117275" name="amazletlink" target="_blank"><img src="https://images-fe.ssl-images-amazon.com/images/I/51%2B3rS3-HPL._SL160_.jpg" alt="Cython ―Cとの融合によるPythonの高速化" style="border: none;" /></a></div><div class="amazlet-info" style="line-height:120%; margin-bottom: 10px"><div class="amazlet-name" style="margin-bottom:10px;line-height:120%"><a href="http://www.amazon.co.jp/exec/obidos/ASIN/4873117275" name="amazletlink" target="_blank">Cython ―Cとの融合によるPythonの高速化</a><div class="amazlet-powered-date" style="font-size:80%;margin-top:5px;line-height:120%">posted with <a href="http://www.amazlet.com/" title="amazlet" target="_blank">amazlet</a> at 19.03.24</div></div><div class="amazlet-detail">Kurt W. Smith <br />オライリージャパン <br />売り上げランキング: 223,491<br /></div><div class="amazlet-sub-info" style="float: left;"><div class="amazlet-link" style="margin-top: 5px"><a href="http://www.amazon.co.jp/exec/obidos/ASIN/4873117275" name="amazletlink" target="_blank">Amazon.co.jpで詳細を見る</a></div></div></div><div class="amazlet-footer" style="clear: left"></div></div>

### feature_importances_の可視化をscikit-learnに組み込む
具体例としてやっていく。

モデルは決定木、データはiris(楽なので)。

#### どういう可視化か
実際に組み込む前に、どういう風に可視化するのか追っていこう。まずはモデルを学習するところまで。

```python3
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_formats = {'png', 'retina'}
import seaborn as sns
sns.set() #デザインのためだけのimport

from sklearn.datasets import load_iris
from sklearn import tree

clf = tree.DecisionTreeClassifier(random_state=42)
iris = load_iris()
clf.fit(iris.data, iris.target)
```
```
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=42, splitter='best')
```

はい、モデルの学習が終わった。
特徴量の重要度を見るには、
```
clf.feature_importances_
```
```
array([0.01333333, 0.        , 0.56405596, 0.42261071])
```
とすればよいが、これをみてどの特徴量の重要度がどれぐらいと即座に分かる人は少ないだろう(特徴量の順番を覚えている超人は除く)。また特徴量の数が多くなったときは、表示が省略され見にくくなる。
そこで重要度を可視化して、見やすくしたい。

そのために以下のようなコードを書いた。
```
importance=pd.Series(data=clf.feature_importances_, index=iris.feature_names)
importance=importance.sort_values()
importance.plot.barh()
```
![](fig/feature_impotances.png)

重要な特徴量から表示するような横向きの棒グラフである。これならば見やすい。

しかし、何回もこれ呼び出そうとすると面倒だ。そこでscikit-learnのDecisionTreeClassifierにメソッドとして組み込んでしまおうというわけである。

#### DecisionTreeClassifierの改変
DecisionTreeClassifierは[この場所で記述されている](https://github.com/scikit-learn/scikit-learn/blob/2718d6212f92220d5f228bfaf7bff0e75ea14965/sklearn/tree/tree.py#L534)

[897行目](https://github.com/scikit-learn/scikit-learn/blob/2718d6212f92220d5f228bfaf7bff0e75ea14965/sklearn/tree/tree.py#L897)に以下のコードを挿入してあげよう。

```python3
    def show_importances(self, feature_names, kwarg={}):
        """show feature importances in this model.

        Parameters
        ----------
        fearure_names : list or array, shape = [n_features]
            feature names sequence when model trained.
        kwarg : keyword arguments of pandas plot
        """
        import pandas as pd
        importance = pd.Series(
            data=self.feature_importances_,
            index=feature_names).sort_values()
        importance.plot.barh(**kwarg)
```

さて、もうすでにscikit-learnは改変された。ちゃんと`pip install --editable .`でインストールしたscikit-learnであるば、`.py`への編集は即時適応される。

今回は行わないが、もし`.pyx`を編集したなら、`python setup.py build_ext --inplace`を実行することも必要だ。

#### 可視化、再訪
もう一度、特徴量重要度の可視化をやりなおしてみよう。まずは先程と同じく訓練終了まで実行する。
```
import numpy as np

%matplotlib inline
import seaborn as sns
sns.set() #デザインのためだけのimport

from sklearn.datasets import load_iris
from sklearn import tree

clf = tree.DecisionTreeClassifier(random_state=42)  # limit depth of tree
iris = load_iris()
clf.fit(iris.data, iris.target)
```

可視化
```
clf.show_importances(iris.feature_names)
```
![](fig/feature_impotances.png)

終わりである。
自作関数をクラスのメソッドとしてきれいにまとめる事ができた。

この具体例なら俺はもっと楽なやり方を知ってるぞ！という方もいると思う(実際ある)。しかし、既存のアルゴリズムを改変する、という作業はこの具体例で流れがつかめたことだろう。

### まとめ
scikit-learnに改変を加えて使うため、必要な導入手順や知識の鱗片を紹介した。
また、最後には具体例として、`feature_importances_`の可視化をメソッドとして`DecisionTreeClassifier`に追加した。

まとめようとすると、また長くなってしまいそうだ。読者がもう一回目次を読んで、どんなことを行ったかを思い出すことをまとめとしよう。

[TOC]








