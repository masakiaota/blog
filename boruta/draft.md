borutaの下書き

## まずは適当に書き出す
### 必要な知識
- ランダムフォレストが特徴量重要度を算出することを知ってる
- なんとなく検定を知っている

### 特徴量選択の必要性
- 特徴量選択(feature selection, 変数選択)は原因分析に使われる。
- kaggleでは判別の精度が重要視されるが、実務上どうしてそのような判別をしたのかが重要である。回帰問題でも同じである。
- 例えば製造業などの場合、欠陥品か否かを見分けるシステムを作るよりも、欠陥品そのものを減らせたほうが良い。そこで欠陥品となる原因を知ることもデータサイエンスに求められている。
- そこで用いられるのが特徴量選択である。

- 特徴量選択した結果、モデルの学習や推論が高速化されるメリットもある。また、判別の精度がよくなったりすることもある。


### borutaとはなにか
- ランダムフォレストと検定を用いた特徴量選択手法の一つ
- 経験上非常に強力
- 判別や回帰の性能が著しく下がったことはない。下がっても誤差の範囲内。
- 元論文
- python実装(ただしバグがあり、動かない。あとで修正したものを提示する。)

###  よく知られた手法との比較
- ランダムフォレスト
    - どれぐらいの特徴量重要度があったら重要だと言えるのかがイマイチ
    - ランダム性から訓練するたびに特徴量重要度が変動する

- Forward selectionやBackward eliminationと言ったステップワイズな方法
    - 選んだ特徴量が過学習する

- lasso
    - 選んだ特徴量が過学習する

### シンプルにアイデア
1. 判別に寄与しないはずの偽の特徴量を作る。
2. 偽の特徴量と一緒にランダムフォレストを訓練。
3. 各特徴量の重要度と偽の特徴量の特徴量を比較。
4. 複数回比較し検定を行うことで、本当に重要な特徴量のみを選択。

以下では判別(回帰)に少しでも寄与するという意味を込めて「重要」と呼ぶことにする。

### もっと詳しく
#### 1. 判別に寄与しないはずの偽の特徴量を作る
(shadow featuresの図を挿入)

1. もともとのDataFrame(Original data)をコピーする。これをShadow featuresと呼ぶことにする。
2. Shadow featuresの各列に対して、サンプルをシャッフルする。これで各特徴量は判別に寄与しないはずの特徴量になった。

このShadow featuresが偽の特徴量となる。

#### 2.偽の特徴量と一緒にランダムフォレストを訓練
Original dataとShadow featuresを結合してランダムフォレストを訓練する入力とする。

#### 3. 各特徴量の重要度と偽の特徴量の重要度を比較。
1. ランダムフォレストを訓練したら、Original dataとShadow featuresの双方から特徴量の重要度を得ることができる。

2. Shadow featuresの中で一番大きな特徴量重要度を記録しておく。この操作によって、「寄与しないはずの特徴量でもこれぐらいの重要度になりえる」という目安になる。

3. ではこの「Shadow featuresの中で一番大きな重要度を持つ特徴量」よりも重要な特徴量が、真に重要だと言ってしまって良いのだろうか。ランダムフォレストの性質により、特徴量の重要度は訓練するたびに変動する。一回の訓練ではたまたま選ばれたり、たまたま選ばれなかったりする特徴量も出てきてしまう。


#### 4. 複数回比較し検定を行うことで、本当に重要な特徴量のみを選択。
そして検定へ。何回もランダムフォレストを訓練し、重要そうな特徴量を複数回記録する。そして各特徴量に対して重要かどうかの検定を行う。

重要そうな特徴量の記録については下図のように非常にシンプルである。
(例の図を挿入)
Shadow featuresの中で最大の重要度よりも大きな特徴量について、それが選ばれるたびhitを+1していく。例えばランダムフォレストを3回訓練し、3回とも選ばれるような特徴量が存在したら、その特徴量のhitには3が格納されている。逆に一回も選ばれなければhitは0である。直感的にはhitが多いほど重要で、少ないほど重要じゃないとなる。

では具体的ランダムフォレストを訓練した回数に対してhitがどれぐらい多ければ重要だと言えるのだろうか。この部分に、ランダムフォレストを訓練した回数をn、p=0.5としたときの二項分布を用いて検定を行う。
詳細は次章で述べる。

### 二項分布による検定

#### 検定の流れ
検定は一般的に以下のような手続きで行われる。詳細と直感的な解釈は後で与えるので、さらっと読んでもらえれば良い。
1. 棄却したい帰無仮説と受容したい対立仮説を用意する。
2. 観測値から検定統計量Tを定める。
3. 帰無仮説が正しいとしてTの分布を求める。
4. 十分小さい有意水準αを定め、帰無仮説が正しいときに$$P(T \in C_\alpha)=\alpha$$となる領域$$C_\alpha$$を棄却域とする。
5. 観測されたTが$$C_\alpha$$に入っていたら対立仮説を受容し、入っていなければ帰無仮説を受容する。

borutaでは以上の手順を各特徴量に対して、一つずつ検定を行っている。ある一つの特徴量(以下、この特徴量)について検定を行うという状況を具体的に見ていこう。

#### 1. 棄却したい帰無仮説と受容したい対立仮説を用意する。
検定を行う際にここでは仮説を3つ用意する。

- 帰無仮説
    - この特徴量の重要度は、判別(回帰)に寄与しない特徴量の重要度と同じである。
- 対立仮説1
    - この特徴量の重要度は、判別(回帰)に寄与しない特徴量の重要度よりも大きい。
- 対立仮説2
    - この特徴量の重要度は、判別(回帰)に寄与しない特徴量の重要度よりも小さい。


#### 2. 観測値から検定統計量Tを定める。
検定統計量は都合のいいように設計できるが、今回hit(重要だとされた回数）がそのまま検定統計量Tとなる。


#### 3. 帰無仮説が正しいとしてTの分布を求める。
帰無仮説は「この特徴量の重要度は、判別(回帰)に寄与しない特徴量の重要度と同じである。」というものだった。
つまり、複数回ランダムフォレストを構築し何度も重要度の勝ち負けを見たときに、「この特徴量と寄与しない特徴量は同じぐらい勝ったり負けたりする。」と言える。勝率0.5である。これを確率分布で表すと二項分布になる。
例えば、一回ランダムフォレストを訓練したときに、この特徴量のhitが0なのか1なのかを確立で表すと、50%50%だろう。これがn=1の場合の二項分布である。図にすると下図のようになる。

（図を挿入）

これを10回勝負にすると下図。勝率0.5なのに、全敗(hit=0)や全勝（hit=10）する確率はほぼないわけである。

(図)

20回勝負ともなるとだいぶ正規分布に近づく。以下、ランダムフォレストを20回訓練した状況について考える。

(図)

結論として考えるべき分布は、ランダムフォレストを訓練した回数をnとしたときの二項分布がhitの従う分布である。

#### 4. 十分小さい有意水準αを定め、帰無仮説が正しいときに$$P(T \in C_\alpha)=\alpha$$となる領域$$C_\alpha$$を棄却域とする。
有意水準αを仮に0.05と設定したとき、棄却域は下図のように設定できる。
(図)

#### 5. 観測されたTが$$C_\alpha$$に入っていたら対立仮説を受容し、入っていなければ帰無仮説を受容する。
今、対立仮説が2つあるので、対応関係を示すとこうなる。
(図)

ランダムフォレストを20回訓練し、hitが18だった場合、いま考えている特徴量はめでたく対立仮説1を受容し、重要だと言えるわけである。

（図）

逆に対立仮説2を受容してしまった特徴量は、計算から省くことでborutaは繰り返し計算をどんどん軽くしている。
また有意水準αは固定ではなく、borutaではBonferroni correctionという手法で決定している。python実装にはBonferroni correctionの代わりに、Benjamini Hochberg FDRを用いることができるが、経験上前者のほうがよい結果を出す。

### 実験
(notebookにて)
