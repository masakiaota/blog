降順リストに対するbisectの実装 list.sort(reverse=True)に対する配列二分法アルゴリズム
===

### はじめに
Pythonにおいて、降順リスト向けの配列二分法アルゴリズムを実装しました。

使用するメリット

- コピペで標準ライブラリに準拠した動作をします。
- 標準ライブラリと異なり、降順リストを扱います。
- 昇順リストに変換し直す計算量と、昇順のidxを降順のidxに変換する思考リソースを削減します


競技プログラミング等ではソートされたリストに対して条件を満たすindexをすばやく返したいことが発生します。
二分探索を用いるとそれをO(logN)で高速に実現できます。

手っ取り早く実装するためにPythonではデフォルトのライブラリでbisectというものが存在します。

https://docs.python.org/ja/3/library/bisect.html

しかし、このライブラリは降順にソートされたlistをサポートしていません。
処理の都合上、`sorted(list, reverse=True)`で返される降順リストをそのまま処理したほうが素直な実装になる場合はしばしばあります。

そこで降順リスト向けの`bisect_left()`と`bisect_right()`を実装しました。


### コピペ用
```python
def bisect_left_reverse(a, x):
    '''
    reverseにソートされたlist aに対してxを挿入できるidxを返す。
    xが存在する場合には一番左側のidxとなる。
    '''
    if a[0] <= x:
        return 0
    if x < a[-1]:
        return len(a)
    # 二分探索
    ok = len(a) - 1
    ng = 0
    while (abs(ok - ng) > 1):
        mid = (ok + ng) // 2
        if a[mid] <= x:
            ok = mid
        else:
            ng = mid
    return ok

def bisect_right_reverse(a, x):
    '''
    reverseにソートされたlist aに対してxを挿入できるidxを返す。
    xが存在する場合には一番右側のidx+1となる。
    '''
    if a[0] < x:
        return 0
    if x <= a[-1]:
        return len(a)
    # 二分探索
    ok = len(a) - 1
    ng = 0
    while (abs(ok - ng) > 1):
        mid = (ok + ng) // 2
        if a[mid] < x:
            ok = mid
        else:
            ng = mid
    return ok
```


テスト的ななにか
```python
#     0  1  2  3  4  5  6   7   8
a = [10, 9, 5, 3, 3, 3, 2, -3, -3]
test = [4, 9, 11, 10, 3, -3, -4]
for t in test:
    print(t, bisect_left_reverse(a, t))
```
```
4 3
9 1
11 0
10 0
3 3
-3 7
-4 9
```

```python
for t in test:
    print(t, bisect_right_reverse(a, t))
```
```
4 3
9 2
11 0
10 1
3 6
-3 9
-4 9
```


ちゃんと動いてるっぽいですね