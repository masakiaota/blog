### 内容3行
- 教師の質が悪い場合でもうまく学習したい。(noisy labelの状況)
- confident learningを使ってみた。RCV1 v2(文章のtf-idf)データセットを使って実験した。
- noisy labelsにPseudo-labelingをしたのが良い結果になった。

### 構成
- はじめに
  - 現実の判別問題において教師が完璧であることは珍しい。このように間違って教師をnoisy labelやcorrupted labelなどという。
  - ICML2020にConfident Learningという論文が投稿された。内容は、noisy labelを取り除くというもの。
  - しかも実装も公開されていたので使ってみた。
  - 結論としてはnoisy labelsを検出した後にそれらについてpseudo labelingするのが良さそうであると判明した。(このデータでは)
  <!-- - 別のデータでも良さそうかは要検証。(参考までに筆者は代表的なtoy ploblemでやっている) -->

- Confident Learning
  - ICML2020に投稿されたnoisy labelを検出する枠組み
  - 判別器にどのようなモデルを用いても良い
  - 詳しくは別の記事を参考
  - cleanlabという実装が存在する

- 実験計画
  - 知りたいのは以下である
    - 真のきれいなラベルで学習させたときの性能 (理想的な性能)
    - noisy labelを用いて学習させたときの性能 (baselineとなる性能)
    - そしてnoisy labelを用いてConfident Learningを適応したときの性能

  - ちょっとバリエーションも入れて具体的には5の実験を行った。(5つは視覚的にわかるように図で説明)
    - ML:clean
    - ML:noisy
    - CL:without noises
    - CL:pseudo for noises
    - CL:pseudo for noises and test

- データセット and 実験設定
  - RCV1 v2データセット
    - reutersの文章をtf-idfにしたもの
    - multi labelのデータセットだが、大分類の4クラスだけ残しsingle labelに加工
    - sampleの数 685071 それぞれのクラスの真のvalue counts [299612  54695 163135 167629]
    - 選んだ理由
      - ラベルがチェック済みであり、noiseがほぼ存在と思われる。そのため評価に適している。 詳しくは[RCV1: A New Benchmark Collection for Text Categorization Research](http://www.jmlr.org/papers/v5/lewis04a.html)を参照。
      - またconfident learningの論文やcleanlabのサンプルでは特徴量が密なデータセットで有効性を検証していたが、要素の97％が0であるような疎なデータセットではどうなるか気になっていた。

  - noiseの生成に用いた分布は以下。noise rateは27%。
```
p(given=i|true=j) =
[[0.68936167 0.         0.         0.        ]
 [0.2387445  0.85410683 0.21184431 0.05112328]
 [0.         0.14589317 0.78815569 0.28050091]
 [0.07189383 0.         0.         0.66837581]]
```
  - 学習について
    - 判別器はLogistic Regression。比較的軽量。
    - パラメータチューンなしの一発勝負(真のラベルがわからない現実では、モデル選択は非常に困難)
    - 4 fold-CVで評価

- 実験結果

| method\accuracy                   | test1      | test2      | test3      | test4      | mean  (std)         |
| --------------------------------- | ---------- | ---------- | ---------- | ---------- | ------------------- |
| ML:clean  (ideal performance)     | 0.9738     | 0.9745     | 0.9745     | 0.9748     | 0.9744 (0.0004)     |
|                                   |            |            |            |            |                     |
| ML:noisy   (baseline performance) | 0.9529     | 0.9529     | 0.9543     | 0.9542     | 0.9536 (0.0007)     |
| CL:wituout noisy labels           | 0.9594     | 0.9594     | 0.9602     | 0.9599     | 0.9598 (0.0004)     |
| CL:pseudo for noises              | 0.9618     | **0.9624** | **0.9628** | **0.9633** | **0.9626** (0.0006) |
| CL:pseudo for noises and test     | **0.9620** | 0.9622     | 0.9627     | 0.9631     | **0.9625** (0.0005) |

ちなみに、どれぐらいnoisy labelを当てられたかと言うと
y_train_corrupted contains 152985 errors. error rate is 27 %

```
             precision    recall  f1-score   support

       False       0.95      0.97      0.96    395071
        True       0.91      0.87      0.89    152985

   micro avg       0.94      0.94      0.94    548056
   macro avg       0.93      0.92      0.92    548056
weighted avg       0.94      0.94      0.94    548056
```

| 評価指標\ | performance |
| --------- | ----------- |
| accuracy  | 0.94        |
| precision | 0.91        |
| recall    | 0.87        |
| f1-score  | 0.89        |



- 実装編
  - 徒然なるままに書こう

- まとめ 
  - ICML2020 Confident Learning の効果を、人工的にノイズを作ることで検証しました。
  - 単純にnoiseを
  - 今回pseudo labelingの手法と組み合わせたように、CLの枠組みは他の手法と組み合わせやすい枠組みである。そしてそうすることで性能が向上する可能性を秘めていることを確認した。例えば, pseudo labeling以外のsemi-supervisedな手法と組み合わせても良さそうだ。



### 各章