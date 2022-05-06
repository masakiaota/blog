kaggle notebookで`pandas.read_pickle`ができない原因と対策法
===

### 背景

最近kaggleを始めて、困ったことに遭遇した。しかも調べても対策法を見つけられなかったため、自分の対策方法を記録に残すことにした。

遭遇した問題は以下のような状況で発生する。

1. kaggle notebook以外の環境で`pandas.DataFrame`を`to_pickle`で保存する。

2. kaggle notebookにて`pandas.read_pickle`をしようとすると、以下のエラーが出て読み込めない。
    - `ValueError: unsupported pickle protocol: 5`
    - `AttributeError: Can't get attribute 'new_block' on <module'pandas.core.internals.blocks'`

本ブログでは原因を解説したあとに手っ取り早い対処方法を紹介する。

解決策だけ知りたい方は、以下の**手っ取り早い対処法**までスキップして頂いて問題ない。


### 原因は二重にある

#### Pythonのversionの問題

kaggle notebookのversionは古くpickle protocol: 5がサポートされていないのが1つ目の原因だ。

`ValueError: unsupported pickle protocol: 5`と怒られているのでprotocol: 5に対応すればいいのである。

kaggle notebookで`import pickle5 as pickle`を実行することでこの問題を解決できる可能性がある。

しかし更に異なるエラーが発生する場合もある。それが次の問題である。


#### Pandasのversionの問題
kaggle notebookのpandasのversionが古く、versionの新しいpandasのobjectと互換性がない。

`AttributeError: Can't get attribute 'new_block' on <module'pandas.core.internals.blocks'`
と怒られているのがpandasの互換性のない部分である。

しかしpandasのversionをあげようにもPython自体のversionが古いため、kaggle notebook上でpandasをこれ以上upgradeする事はできない。

要はkaggle notebookの方のversionを合わせてpickle化した`pandas.DataFrame`を読み込むことは非常に困難というわけである。

### 手っ取り早い対処法

#### 自分の作業環境側
1. `pandas.DataFrame.to_dict()`で一回辞書型に変換する。
2. joblibを用いて保存(joblibでなくても可)する。

#### kaggle notebook側
1. joblibで保存した辞書を読み込み直す。
2. `pandas.DataFrame(辞書)`で再びpandasで取り扱えるように変換する。


### サンプルコード
#### 自分の作業環境側
```python
tmp=df.to_dict() #辞書に変換
joblib.dump(tmp, ファイル名, compress=3) #辞書をjoblibで保存
```


#### kaggle notebook側
```python
tmp=joblib.load(ファイル名) #辞書を読み込み
df=pd.DataFrame(tmp) #再度DataFrameに変換
```

### 根本的な対処法
自分のkaggle用環境では、kaggle notebookと環境をなるべく揃えましょうという教訓だった。

### まとめ

本ブログの内容を大雑把に箇条書きで俯瞰し締めようと思う。

- 背景
    - kaggle notebook以外の環境で`pandas.DataFrame`を`to_pickle`で保存する。
    - kaggle notebookにて`pandas.read_pickle`をしようとすると、エラーが出て読み込めない。
- 原因は二重にある
    - Pythonのversionの問題
    - Pandasのversionの問題
- 手っ取り早いの対処法
    - 自分の作業環境側
        1. `pandas.DataFrame.to_dict()`で一回辞書型に変換する。
        2. joblibを用いて保存(joblibでなくても可)する。
    - kaggle notebook側
        1. joblibで保存した辞書を読み込み直す。
        2. `pandas.DataFrame(辞書)`で再びpandasで取り扱えるように変換する。
    - サンプルコードも示した。
- 根本的解決法
    - kaggle notebookと環境をなるべく揃えましょう。

おしまい。