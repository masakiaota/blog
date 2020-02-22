めぐる式二分探索 コピペで使えるPython実装
===

### はじめに
AtCoderで二分探索を実装するときバグらせないように考えると結構時間かかりませんか？自分はかかります。

競技プログラミング界隈ではめぐる式二分探索という二分探索の書き方(流派？)があり、使いやすい、バグりにくくなど様々なメリットがあります。

Python実装を公開しているブログはパット見、見つからなかったのでおいておきます。使用例もおいておきます。

### めぐる式二分探索のメリットと参考文献
めぐる式二分探索を使うメリットとして以下があげられます。
- 配列化できない関数を探索可能 (bisectモジュールでは不可)
- バグりにくい (終了状態がきちんとしている)
- ライブラリとして扱うことが可能で実装が高速化される
- 思考リソースの消耗を防げる (条件を満たすかそうでないかだけ考えれば良い)

仕組みについては別の文献に任せます。

https://twitter.com/meguru_comp/status/697008509376835584

https://qiita.com/drken/items/97e37dd6143e33a64c8c

### コピペ用
```python
def is_ok(arg):
    # 条件を満たすかどうか？問題ごとに定義
    pass


def meguru_bisect(ng, ok):
    '''
    初期値のng,okを受け取り,is_okを満たす最小(最大)のokを返す
    まずis_okを定義すべし
    ng ok は  とり得る最小の値-1 とり得る最大の値+1
    最大最小が逆の場合はよしなにひっくり返す
    '''
    while (abs(ok - ng) > 1):
        mid = (ok + ng) // 2
        if is_ok(mid):
            ok = mid
        else:
            ng = mid
    return ok
```


### 例題

ABC146のC問題からの出題。

https://atcoder.jp/contests/abc146/tasks/abc146_c?lang=ja

整数屋さんで整数を買えるかどうかを二分探索します。

今、整数を買える状態をTrueとしたいので、`is_ok`は以下のように定義すれば良いです。

```python
def is_ok(arg):
    # 整数を買えればTrueを返す
    return A * arg + B * len(str(arg)) <= X
```

これで準備は完了です。あとは`meguru_bisect`の引数を入れるだけです。条件を満たさない(ng)整数の最大は制約より`10**9`で、条件を満たす(ok)の最小は`1`です。これらは答えになることもあり得るので、引数にはその範囲外を指定してください。つまり`meguru_bisect(ng=10**9+1, ok=0)`です。

これを踏まえると答えは以下のようになります。

```python
A, B, X = map(int, input().split())

def is_ok(arg):
    # 整数を買えればTrueを返す
    return A * arg + B * len(str(arg)) <= X

def meguru_bisect(ng, ok):
    while (abs(ok - ng) > 1):
        mid = (ok + ng) // 2
        if is_ok(mid):
            ok = mid
        else:
            ng = mid
    return ok

print(meguru_bisect(10**9 + 1, 0))
```


別問題 
https://atcoder.jp/contests/arc037/tasks/arc037_c

bisectとめぐる式のあわせ技で比較的軽量に実装できます(詳しくは解説をネットでググってください)。

okとngの大小関係が先程と逆になっていますが、問題なく動きます。

```python
from bisect import bisect_right
import sys
read = sys.stdin.readline
 
def read_ints():
    return list(map(int, read().split()))
 
N, K = read_ints()
A = read_ints()
B = read_ints()
A.sort()
B.sort()
 
def is_ok(X):
    # a_i * b_j <= Xを満たす個数がKよりも多いか？
    cnt = 0
    for a in A:
        aa = X // a
        cnt += bisect_right(B, aa)
    return cnt >= K
 
def meguru_bisect(ng, ok):
    while (abs(ok - ng) > 1):
        mid = (ok + ng) // 2
        if is_ok(mid):
            ok = mid
        else:
            ng = mid
    return ok
  
print(meguru_bisect(-1, 10 ** 18 + 1))
```
