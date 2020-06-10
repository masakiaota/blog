蟻本Python回答集 中級前編 (P127~P187)
===

### はじめに
AtCoder青を目指しつつデータ構造など勉強するため、プログラミングコンテストチャレンジブック [第2版] [■](https://book.mynavi.jp/ec/products/detail/id=22672) (通称、蟻本)を解くことした。
せっかくなのでPythonでの解答をここに記録する。

Pythonで解答してる人のブログを漁っても初級編の途中(DPとか)で挫折してる人が多そうだったので誰かの助けになれたらと思う。

著作権保護のため本の内容をすべて公開するわけではないので、解説などは本を見てほしい。
一方、変数名の説明なしにコード例が書いてある問題もいくつかあり、そういう問題はコードのコメントに意味合いを補足した。

この本の購入に関してはmynavi booksからPDFを購入するのがおすすめだ。ノートアプリを使っていろいろ書き込むことができる。

初級編はこちら

https://aotamasaki.hatenablog.com/entry/2020/05/17/ants_book_part1

### 章 3-1 値の検索だけじゃない！”二分探索”
最小値の最大化、最大値の最小化、平均値の最大(小)化に二分探索を使う問題は緑~水色の問題にちょくちょく出てくる気がする。この章はかなり役になった。

#### P128 lower_bound
```python
n = 5
a = [2, 3, 3, 5, 6]
k = 3

from bisect import bisect_left
print(bisect_left(a, k))
```

#### P129 Cable master
めぐる式二分探索の作法で実装する。
```python
N = 4
K = 11
L = [8.02, 7.43, 4.57, 5.39]

# K本以上作れる長さxの内、最大のxを探索する
def is_ok(x):
    n = 0  # 何本長さxの紐が作れるか
    for l in L:
        n += l // x
    return n >= K

def meguru_bisect(ng, ok):
    while abs(ok - ng) > 10 ** -3:  # 10^-3の誤差は許容される
        mid = (ok + ng) / 2
        if is_ok(mid):
            ok = mid
        else:
            ng = mid
    return ok


print(meguru_bisect(10**6, 0.1))  # 二桁を出力しろだが本質ではないので全部出力しちゃう
```

#### P131 Aggressive cows

```python
# 牛の座標をciとするとmax_{配置} min_i (c_{i+1}-c_i) を求める問題
# 牛を配置したときに最小の距離がmin_i (c_{i+1}-c_i)がdであるとしたときに、矛盾することなく牛を並べることができる最大のd
# と問題を言い換えられる。
N = 5
M = 3
X = [1, 2, 8, 4, 9]

X.sort()
def is_ok(d):
    # 間隔d以上で牛を並べることができればTrue
    nex = -10000
    n = 0  # 並んだ牛の数
    for x in X:
        if x >= nex:
            n += 1
            nex = x + d
    return n >= M


def meguru_bisect(ng, ok):
    while (abs(ok - ng) > 1):
        mid = (ok + ng) // 2
        if is_ok(mid):
            ok = mid
        else:
            ng = mid
    return ok

print(meguru_bisect(10**9 + 1, -1))
```

#### P132 平均最大化
```python
# max_{i=i_1,i_2,...,i_k} (Σvi/Σwi) となるようなiの選び方
# これも答えをxと仮定すると、x以上となる選び方が存在する中の最大のx
# Σ(vi-x*wi) >= 0 となるので greedyにk個選んだときに条件を満たすことができるか判別できる
# 計算量は O(NlogNlog(max x))

N = 3
K = 2
W = [2, 5, 2]
V = [2, 3, 1]


def is_ok(x):
    # 単位価値がx以上となる選び方は存在するか？
    VXW = [vi - x * wi for vi, wi in zip(V, W)]
    VXW.sort(reverse=True)
    return sum(VXW[:K]) >= 0

def meguru_bisect(ng, ok):
    while (abs(ok - ng) > 10**-9):  # 小数8桁ぐらいの精度は保証する
        mid = (ok + ng) / 2
        if is_ok(mid):
            ok = mid
        else:
            ng = mid
    return ok

print(meguru_bisect(10**6 + 1, 0))
```

### 章 3-2 厳選！ 頻出テクニック(1)

#### P135 Subsequence
総和がS以上→総和がSを未満の範囲の長さ+1と言い換える

```python
# この問題を見たらしゃくとり法よりも累積和を使いたくなるな

def two_pointers(ls: list, S):
    n_ls = len(lsS
    ret = []

    r = 0
    s = 0
    for l in range(n_ls):
        while r < n_ls and s + ls[r] < S:  # 初めて条件を満たす一歩手前をr)にする。
            s += ls[r]
            r += 1
        ret.append((l, r))
        if r == n_ls:
            break
        # 抜けるときの更新
        s -= ls[l]
    return ret


def solve(n, S, A):
    idxs = two_pointers(A, S)
    print(idxs)
    if len(idxs) == 1:
        print(0)  # S以上にすることはできない
    else:
        print(min([r - l + 1 for l, r in idxs]))


# 入力例1
n = 10
S = 15
A = [5, 1, 3, 5, 10, 7, 4, 9, 2, 8]
solve(n, S, A)

# 入力例2
n = 5
S = 11
A = [1, 2, 3, 4, 5]
solve(n, S, A)

# 入力例 オリジナル (すべてがS以上)
n = 5
S = 1
A = [3, 2, 3, 4, 5]
solve(n, S, A)

# 入力例 オリジナル (S以上になれない)
n = 5
S = 100
A = [1, 2, 3, 4, 5]
solve(n, S, A)

# 入力例 オリジナル (ちょうどS)
n = 5
A = [1, 2, 3, 4, 5]
S = sum(A)
solve(n, S, A)
```


#### P137 Jessica's Reading Problem
```python
P = 5
A = [1, 8, 8, 1, 1]  # ちょっと 蟻本のサンプルと違うけど

n = len(set(A))  # ユニークの種類数

from collections import Counter
cnt = Counter([])  # default dict代わり

idxs = []
r = 0
num = 0  # 種類数
for l in range(P):
    while r < P and num + (cnt[A[r]] == 0) < n:  # 初めて条件を満たす一歩手前をr)にする。
        if cnt[A[r]] == 0:
            num += 1
        cnt[A[r]] += 1
        r += 1
    idxs.append((l, r))
    if r == P:
        break  # これ以上短くしても条件を満たすことはない
    # 抜けるときの更新
    if cnt[A[l]] == 1:
        num -= 1
    cnt[A[l]] -= 1


print(min([r - l + 1 for l, r in idxs]))
```

#### P139 Face The Right Way
ここから反転テクとなる。本質的には同じだが、ここではimos法を使って反転を管理する。
(imos法はググってね)
```python
# Kについて全探索する必要あり
# 一番左からgreedyに反転していけばいい(?)
# だけどKの探索にO(N)、区間反転がO(N)で、区間の反転にO(N)かかることからO(N^3)かかってしまう。
# 区間反転は高速化できる！

N = 7
cows = 'BBFBFBB'
ans_M = 10**5
ans_K = 10**5
for k in range(1, N + 1):  # Kについて全探索
    m = 0
    is_valid = True  # 有効なkか？
    is_fliped = [0] * (N + 1)
    for i in range(N):  # 各牛について左から見ていく
        is_fliped[i] += is_fliped[i - 1]  # デルタ関数を積分してステップ関数を作るイメージ
        if is_fliped[i] & 1:  # 奇数のときは反転してる
            if cows[i] == 'B':
                continue
        else:  # ひっくり返ってない牛
            if cows[i] == 'F':
                continue
        # ひっくり返す作業が必要
        m += 1
        # K個ひっくり返す #ここではデルタ関数を立てるイメージ
        if i + k > N:
            is_valid = False  # ピッタリ牛をひっくり返すことはできない！
            break
        is_fliped[i] += 1
        is_fliped[i + k] -= 1

    if is_valid:
        print(k, m, is_fliped)  # 確認用
        if m < ans_M:
            ans_M = m
            ans_K = k
print(ans_K, ans_M)
```

#### P141 Fliptile
```python
# 重要な考察 : 現在位置から1マス上が黒になる場合、現在位置を必ず踏む必要がある
# 詳しい話は蟻本で

# 主に必要なデータと関数
# 1. あるマスについて踏んだかどうかを示す配列
# 2. 踏んだかどうかの情報からそのマスが黒か判断する関数
# 3. 1(0)行目の踏み方を全探索する関数
# 4. 牛の踏み方が決まったときに何回踏んだかを返す関数

# 入力

M = 4
N = 4
tile = [[1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1]]


from itertools import product
from copy import deepcopy


opt = None  # 最適な盤面の保存
min_flip = 2**31  # 最小のひっくり返す回数


def get_color(x, y, flip):  # 踏んだかどうかの情報からそのマスが黒か判断する関数
    c = tile[x][y]
    # 周りの踏んだ状況を取得
    for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < M and 0 <= ny < N:
            c ^= flip[nx][ny]
    return c


def generate_flip_0row():  # 辞書順で0行目の踏み方をbit全探索するやつ
    for ret in product(range(2), repeat=M):
        yield list(ret)


def simulate(flip):  # 最適な踏み方をシミュレートする
    # flipはすでに0行目が埋まっている前提
    for i, j in product(range(1, M), range(N)):
        if get_color(i - 1, j, flip):  # もし上のタイルが黒ならこのタイルは踏まないと上のタイルを白にできない
            flip[i][j] = 1
    # 有効な踏み方か？つまりM-1行目がすべて白になっているかチェック
    for j in range(N):
        if get_color(M - 1, j, flip):
            return -1  # もし黒があれば強制的に終了

    return sum([sum(x) for x in flip])  # flipした回数


# 実装する
for zeroth_row in generate_flip_0row():
    flip = [[0] * N for _ in range(M)]  # 踏んだマス
    flip[0] = zeroth_row
    tmp = simulate(flip)
    if tmp != -1 and tmp < min_flip:
        opt = deepcopy(flip)
        min_flip = tmp

if opt == None:
    print('IMPOSSIBLE')
else:
    print(*opt, sep='\n')
```

#### P145 Physics Experiment
```python
# R=0のときは解説通り簡単
# R>0のときは以下のように考えると良いだろう
# 例えばボールが2つだった場合、2つ目(1番目)のボールはH+2Rの高さから落ちて,2Rの高さで反射する運動をすると捉えることができる。
# ∵2つのボールが衝突する瞬間、2Rの距離だけ瞬間移動してすり抜けた(位置をswap)と考えることができるから。
# よってすべてのボールについて高さHで計算しておいて、あとで高さ分足しても問題ない

from math import sqrt


def y(t, H):  # 高さH落としたときのt秒後の高さy (弾性衝突)
    g = 10
    tau = sqrt(2 * H / g)  # Hから地面までの落下時間
    if t // tau % 2 == 0:  # 落ちているとき
        t %= tau
    else:  # 反射しているとき
        # 地面からの初速度を考えてもいいけどせっかくなので線対称の関係をうまく使う
        t = tau - t % tau
    return H - 1 / 2 * g * t ** 2


def solve(N, H, R, T):
    ans = []  # 各ボールの座標
    for i in range(N):
        ans.append(y(T - i, H) + 2 * R * i / 100)  # 何気にR`cm`なのでmに直す。
    ans.sort()
    print(*ans)


# 入力例1
solve(1, 10, 10, 100)  # ok
# 入力例2
solve(2, 10, 10, 100)  # ok
```

#### P147 4 Value whose Sum is 0
半分全列挙のテクニックもちょくちょく使いそう。基本的なアイデアは2つの配列に成るまで全列挙して、それらを二分探索で最後に高速化するという枠組み。

```python
from itertools import product
from bisect import bisect_left, bisect_right

n = 6
A = [-45, -41, -36, -36, 26, -32]
B = [22, -27, 53, 30, -38, -54]
C = [42, 56, -37, -75, -10, -6]
D = [-16, 30, 77, -46, 62, 45]


AB = [a + b for a, b in product(A, B)]
CD = [c + d for c, d in product(C, D)]
CD.sort()

ans = 0
for ab in AB:
    ans += bisect_right(CD, -ab) - bisect_left(CD, -ab)  # =abとなる個数
print(ans)
```

#### P148 巨大ナップサック
DPで解けないナップサックも、半分全列挙で解ける可能性がある

```python
# https://onlinejudge.u-aizu.ac.jp/courses/library/7/DPL/1/DPL_1_H
# ここでverifyできる がpythonが遅いためTLEになる...

from itertools import product
from collections import defaultdict
from typing import Dict
from bisect import bisect_left, bisect_right

N, W = map(int, input().split())

VW = []
for _ in range(N):  # 読み込み
    v, w = map(int, input().split())
    VW.append((v, w))

# # 入力例
# N, W = 4, 5
# VW = [(3, 2),
#       (2, 1),
#       (4, 2),
#       (2, 3)]

WV1 = defaultdict(lambda: -1)  # N//2だけ半分全列挙する(w_sumが同じ時はv_sumの大きい方の値を採用する)
N_half = N // 2
for bit in product(range(2), repeat=N_half):
    v_sum, w_sum = 0, 0
    for idx, to_use in enumerate(bit):
        if to_use:
            v, w = VW[idx]
            v_sum += v
            w_sum += w
    if w_sum <= W:
        WV1[w_sum] = max(WV1[w_sum], v_sum)

WV2 = defaultdict(lambda: -1)  # N-N//wの半分全列挙
for bit in product(range(2), repeat=(N - N_half)):
    v_sum, w_sum = 0, 0
    for idx, to_use in enumerate(bit, start=N_half):
        if to_use:
            v, w = VW[idx]
            v_sum += v
            w_sum += w
    if w_sum <= W:
        WV2[w_sum] = max(WV2[w_sum], v_sum)


def to_increase(WV: Dict[int, int]):
    '''WV1,WV2双方をw,vについて真に単調増加にする
    ∵あるW'以下の最大値が知りたい。逆に言うとwが増えて価値が減少するような詰め込み方はいらない'''
    ret = []
    ma = -1
    for w, v in sorted(WV.items()):
        if v <= ma:
            continue
        ma = v
        ret.append((w, v))
    return ret


WV1 = to_increase(WV1)
WV2 = to_increase(WV2)
# 二分探索ですばやくW以下となるVの最大を取得
# print(WV1, WV2)
W2, V2 = zip(*WV2)
ans = -1
for w, v in WV1:
    idx = bisect_right(W2, W - w) - 1  # W-w以下も含めるためのright
    # print(w, v, W2[idx], V2[idx])
    ans = max(ans, V2[idx] + v)

print(ans)
```

#### P150 領域の個数
x,yは独立に座標圧縮できる。ちなみにP151の右図は間違っていて、解説の説明やプログラム通りにシミュレーションしてもこの図にはならないので注意。でもアイデアを伝えるには十分。

```python
from itertools import product

class Compress:  # すべての情報を残すのが好きなのでクラス化した
    def __init__(self, ls):
        self.i_to_orig = sorted(set(ls))
        self.orig_to_i = {}
        for i, zahyou in enumerate(self.i_to_orig):
            self.orig_to_i[zahyou] = i
        self.len = len(self.i_to_orig)

    def __len__(self):
        return len(self.i_to_orig)


w, h, n = 10, 10, 5
x1 = [1, 1, 4, 9, 10]
x2 = [6, 10, 4, 9, 10]
y1 = [4, 8, 1, 1, 6]
y2 = [4, 8, 10, 5, 10]
x1.extend([0, 0, w + 1, w + 1])  # 周りを黒線で囲っておく
y1.extend([0, h + 1, h + 1, 0])
x2.extend([0, w + 1, w + 1, 0])
y2.extend([h + 1, h + 1, 0, 0])

# 必要な座標の確保
# 端点の座標とその周囲-1,+1は確保する
X_comp = Compress([a + d for a, d in product(x1 + x2, (-1, 0, 1))])
Y_comp = Compress([a + d for a, d in product(y1 + y2, (-1, 0, 1))])


# 圧縮済みgridの用意
grid = [[0] * len(Y_comp) for _ in range(len(X_comp))]
# 実際にgridを塗る
for xs, ys, xt, yt in zip(x1, y1, x2, y2):
    # 圧縮後の座標に変換
    xs = X_comp.orig_to_i[xs]
    ys = Y_comp.orig_to_i[ys]
    xt = X_comp.orig_to_i[xt]
    yt = Y_comp.orig_to_i[yt]
    if xs > xt:
        xs, xt = xt, xs
    if ys > yt:
        ys, yt = yt, ys
    # 塗る
    for x in range(xs, xt + 1):
        for y in range(ys, yt + 1):
            grid[x][y] = 1

print(*grid, sep='\n')  # 確認用

# このgridの領域をあとはカウントすればok!
# 領域の個数は最大250*250の62500個
# 圧縮後のgridの要素数は最大3000*3000(9e6)で再帰関数でも多分できるけど、ちょっと怪しい
# がbfsを書くのも面倒なのでdfsで

def dfs(x, y):  # 周囲を探索しながら0に置き換える
    # 終了条件はなくても勝手に止まる
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < len(X_comp) and 0 <= ny < len(Y_comp)):
            continue
        if grid[nx][ny] == 0:
            grid[nx][ny] = 1
            dfs(nx, ny)


ans = 0
for x, y in product(range(len(X_comp)), range(len(Y_comp))):
    if grid[x][y] == 1:
        continue
    else:
        ans += 1
        dfs(x, y)

print(ans - 1)  # 周囲をグルっと囲む0の分、1を引く
```

### 章 3-3 様々なデータ構造を操ろう
#### P156 Crane
いきなり高度な問題。

```python
# 始点と終点のベクトルを更新していくイメージ

from math import sin, cos, pi

class SegmentTree:
    def __init__(self, ls: list, segfunc, identity_element):
        self.ide = identity_element
        self.func = segfunc
        self.n_origin = len(ls)
        self.num = 2 ** (self.n_origin - 1).bit_length()  # n以上の最小の2のべき乗
        self.tree = [self.ide] * (2 * self.num - 1)  # −1はぴったりに作るためだけど気にしないでいい
        for i, l in enumerate(ls):  # 木の葉に代入
            self.tree[i + self.num - 1] = l
        for i in range(self.num - 2, -1, -1):  # 子を束ねて親を更新
            self.tree[i] = segfunc(self.tree[2 * i + 1], self.tree[2 * i + 2])

    def __getitem__(self, idx):  # オリジナル要素にアクセスするためのもの
        if isinstance(idx, slice):
            start = idx.start if idx.start else 0
            stop = idx.stop if idx.stop else self.n_origin
            l = start + self.num - 1
            r = l + stop - start
            return self.tree[l:r:idx.step]
        elif isinstance(idx, int):
            i = idx + self.num - 1
            return self.tree[i]

    def update(self, i, x):
        '''
        i番目の要素をxに変更する(木の中間ノードも更新する) O(logN)
        '''
        i += self.num - 1
        self.tree[i] = x
        while i:  # 木を更新
            i = (i - 1) // 2
            self.tree[i] = self.func(self.tree[i * 2 + 1],
                                     self.tree[i * 2 + 2])

    def query(self, l, r):
        '''区間[l,r)に対するクエリをO(logN)で処理する'''
        if r <= l:
            return ValueError('invalid index (l,rがありえないよ)')
        l += self.num
        r += self.num
        res_right = []
        res_left = []
        while l < r:  # 右から寄りながら結果を結合していくイメージ
            if l & 1:
                res_left.append(self.tree[l - 1])
                l += 1
            if r & 1:
                r -= 1
                res_right.append(self.tree[r - 1])
            l >>= 1
            r >>= 1
        res = self.ide
        # 左右の順序を保って結合
        for x in res_left:
            res = self.func(x, res)
        for x in reversed(res_right):
            res = self.func(res, x)
        return res

# セグ木の各要素は(vx,vy,ang)を持つことにする。angはそのベクトルの右側の辺が垂直から何度傾いているかを示す

def segfunc(l, r):
    # ベクトル同士の結合は以下で定義できる
    c = cos(l[2])
    s = sin(l[2])
    return (l[0] + (c * r[0] - s * r[1]), #vx
            l[1] + (s * r[0] + c * r[1]), #vy
            l[2] + r[2]) # ang


def solve(N, C, L, S, A):
    tmp = [(0, y, 0) for y in L]
    # print(tmp)
    segtree = SegmentTree(tmp, segfunc, identity_element=(0, 0, 0))
    S = [s - 1 for s in S]
    A = [(a - 180) * (pi / 180) for a in A]  # ラジアンに直しておく
    # print(segtree.tree)
    for i, a in zip(S, A):
        x, y, _ = segtree[i]
        segtree.update(i, (x, y, a))
        ansx, ansy, _ = segtree.query(0, N)
        print(ansx, ansy)


# 入力例1
N = 2
C = 1
L = [10, 5]
S = [1]
A = [90]
solve(N, C, L, S, A)

print()

# 入力例2
N = 3
C = 2
L = [5, 5, 5]
S = [1, 2]
A = [270, 90]
solve(N, C, L, S, A)
```

#### P162 バブルソートの交換回数
これはセグメント木でも実装できる。BITを使うメリットとしては実装が楽な点と定数倍が高速なことが挙げられる。

```python
class BIT:
    def __init__(self, n):
        self.n = n
        self.bit = [0] * (self.n + 1)  # bitは(1based indexっぽい感じなので)
    def init(self, ls):
        assert len(ls) <= self.n
        # lsをbitの配列に入れる
        for i, x in enumerate(ls):  # O(n log n 掛かりそう)
            self.add(i, x)
    def add(self, i, x):
        '''i番目の要素にxを足し込む'''
        i += 1  # 1 based idxに直す
        while i <= self.n:
            self.bit[i] += x
            i += (i & -i)
    def sum(self, i, j):
        '''[i,j)の区間の合計を取得'''
        return self._sum(j) - self._sum(i)
    def _sum(self, i):
        '''[,i)の合計を取得'''
        # 半開区間なので i+=1しなくていい
        ret = 0
        while i > 0:
            ret += self.bit[i]
            i -= (i & -i)
        return ret


n = 4
A = [3, 1, 4, 2]
bit = BIT(max(A) + 1)  # 0~Aの最大までの座標を用意しておく
ans = 0
# i<jにおいて,ai>ajとなる要素の個数をカウント
# jを固定すれば、jより前に出現したajよりも大きい要素の数になる
for a in A:
    ans += bit.sum(a + 1, bit.n)  # aより大きい要素の個数
    bit.add(a, 1)

    # for a in range(0, max(A) + 1):
    #     # ちゃんと各要素がなってるかの検証
    #     print(bit.sum(a, a + 1), end=' ')
    # print()

print(ans)
```

#### P163 A Simple Problem with Integer
入力例がないので、省略する。遅延セグ木で解けるので必要ならばググれば良い。

#### P168 K-th Number
区間を分割しておくと計算量が落ちる問題。効率的な区間の分割は√n (平方分割)。

```python
# k番目の数は？→x以下の数がk個以上存在する最小のx を探索
# 区間クエリに答えるには→ √Nごとに区切ったバゲットをソートしておけば都合が良い
# 良い都合→ある区間の処理ははたかだかO(√N logn)でx以下の数の個数を取得できる ∵バゲット内では二分探索で、要素はそのまま見て、x以下の個数を得られる

from typing import List
from math import sqrt
from bisect import bisect_right

n = 7
m = 3
A = [1, 5, 2, 6, 3, 7, 4]
query = [(2, 5, 3), (4, 4, 1), (1, 7, 3)]


B = int(sqrt(n)) + 1  # bucketのサイズ

bucket: List[List[int]] = [[] for _ in range(B)]  # Aのbucket. 各要素はソートされた数列
for i in range(n):
    # print(i // B, i)
    bucket[i // B].append(A[i])

for i in range(len(bucket)):
    bucket[i].sort()


def solve_query(i, j, k):
    # 二分探索でk番目の数字を探索する

    l, r = i - 1, j  # 0basedindex そして 半開区間にする
    l_bucket = (l - 1) // B + 1  # bucketのidx
    r_bucket = r // B  # 半開区間なのでこれでいい

    def is_ok(x):
        # x以下の数がk個以上ならok
        num_el_x = 0
        num_el_x += sum([xx <= x for xx in A[l:l_bucket * B]])
        num_el_x += sum([xx <= x for xx in A[r_bucket * B:r]])
        for i in range(l_bucket, r_bucket):
            num_el_x += bisect_right(bucket[i], x)
        return num_el_x >= k

    def meguru_bisect(ng, ok):
        while (abs(ok - ng) > 1):
            mid = (ok + ng) // 2
            if is_ok(mid):
                ok = mid
            else:
                ng = mid
        return ok

    print(meguru_bisect(-10**9 - 1, 10**9 + 1))

for i, j, k in query:
    solve_query(i, j, k)
```

### 章 3-4 動的計画法を極める！
#### P173 巡回セールスマン問題
PASTで出たときにまだこのページやっていなくて解けなかった...

```python
# https://www.slideshare.net/hcpc_hokudai/advanced-dp-2016 動的計画法の問題の解説がされている 神
# これが比較的わかりやすいかも https://algo-logic.info/bit-dp/

'''
定式化(本は"集める"DPで定義してるが、わかりやすさのため"配る"DPで定式化)

ノーテーション
S ... 頂点集合
| ... 和集合演算子
dp[S][v] ... 重みの総和の最小。頂点0から頂点集合Sを経由してvに到達する。

更新則
dp[S|{v}] = min{dp[S][u]+d(u,v)} ただしv∉S

初期条件
dp[∅][0] = 0 #Vはあらゆる集合
dp[V][u] = INF #ほかはINFで初期化しておく

答え
dp[すべての要素][0] ... 0からスタートしてすべての要素を使って最後に0に戻るための最小コスト
'''

# verify https://onlinejudge.u-aizu.ac.jp/courses/library/7/DPL/all/DPL_2_A
INF = 2 ** 31
from itertools import product

def solve(n, graph):
    '''nは頂点数、graphは隣接行列形式'''
    max_S = 1 << n  # n個のbitを用意するため
    dp = [[INF] * n for _ in range(max_S)]
    dp[0][0] = 0
    for S in range(max_S):
        for u, v in product(range(n), repeat=2):
            if (S >> v) & 1 or u == v:  # vが訪問済みの場合は飛ばす
                continue
            dp[S | (1 << v)][v] = min(dp[S | (1 << v)][v],
                                      dp[S][u] + graph[u][v])

            # # 別解 #集めるDPの発想
            # if u == v or (S >> v) & 1 == 0:  # Sにはvが含まれている必要がある
            #     continue
            # dp[S][v] = min(dp[S][v],
            #                dp[S - (1 << v)][u] + graph[u][v])
    print(dp[-1][0] if dp[-1][0] != INF else -1)


# # 入力例
# n = 5
# graph = [[INF, 3, INF, 4, INF],
#          [INF, INF, 5, INF, INF],
#          [4, INF, INF, 5, INF],
#          [INF, INF, INF, 0, 3],
#          [7, 6, INF, INF, INF]]
# solve(n, graph)


# verify用
n, e = map(int, input().split())
graph = [[INF] * n for _ in range(n)]
for _ in range(e):
    s, t, d = map(int, input().split())
    graph[s][t] = d
solve(n, graph)
```

#### P175 Traveling by Stagecoach
```python
# なんかチケットの使い方(8!)の全探索でダイクストラ法を使えば行けそうだが...
# ここは解説どおりDAGで解く
# 調べると拡張ダイクストラとかいう方法が出てくる
# やってることは拡張ダイクストラと同じだけど、巡回路がないからダイクストラじゃなくても解けるってことか

from collections import deque, defaultdict
# 入力例
n = 2
m = 4
a = 2 - 1  # 0 based indexに変換
b = 1 - 1
t = [3, 1]

road = {0: [(2, 3), (3, 2)],  # from:[(to, cost),...]
        1: [(2, 3), (3, 5)],
        2: [(0, 3), (1, 3)],
        3: [(0, 2), (1, 5)]}


# verify用 ただしMLEになる
# http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=1138&lang=jp
# n, m, p, a, b = map(int, input().split())
# a, b = a - 1, b - 1
# t = list(map(int, input().split()))
# road = defaultdict(lambda: [])
# for _ in range(p):
#     x, y, z = map(int, input().split())
#     road[x - 1].append((y - 1, z))
#     road[y - 1].append((x - 1, z))

# 実装
INF = 2 ** 31  # 2147483648 > 10**9
S_max = pow(2, n)
D = [[INF] * S_max for _ in range(m)]  # D[v][S]...乗車券の状態がSでノードvにたどり着ける時間の最小
D[a][S_max - 1] = 0  # 乗車券を使わずにaにいることは可能

# 幅優先探索ながらDを埋めてく
que = deque([(a, S_max - 1, 0)])  # (現在いるノード、乗車券の状態、そこまでの時間)
while que:
    v, S, time = que.popleft()
    for to, cost in road[v]:
        for i in range(n):
            if (S >> i) & 1 == 0:
                continue  # i番目の乗車券は使えないので処理しない
                # 乗車券が使えなくなれば自動的にwhileが止まる
            S_new = S - (1 << i)
            time_new = time + (cost / t[i])
            que.append((to, S_new, time_new))
            D[to][S_new] = min(D[to][S_new], time_new)

# print(*D, sep='\n')
# bにたどり着くための最小コストの取得
ans = INF
for S in range(S_max):
    ans = min(ans, D[b][S])

print(ans if ans != INF else 'Impossible')  # ok
```

#### P177 ドミノ敷き詰め
```python
# 解説→https://www.slideshare.net/hcpc_hokudai/advanced-dp-2016 18ページから

'''
蟻本と違って
dp[i][j][S]...パターンの総数。(i,j)マスまで埋めたときに、境界(埋めたマスの1つ下)がSになる場合の。

更新則
(i,j)に埋めることができないとき(例えば黒マスになってるとか、すでに埋まってるとか)
dp[i][j+1][S & ~(1<<j)] = dp[i][j][S] ∵i,jに置かない→次の境界のjbit目は必ず空白(and処理で必ず空白にする)
# S & ~(1<<j)はbin(((1<<10) - 1) & ~(1<<5))を実行してみれば正しく動作していることが確認できる。

(i,j)に縦置きを埋めるとき:
    改行が必要ないとき:
        dp[i][j+1][S|(1<<j)] += dp[i][j][S] ∵jbitは必ず埋まる
    改行が必要(つまりj==W-1のとき):
        dp[i+1][0][S|(1<<j)] += dp[i][j][S] ∵jbitは必ず埋まる


(i,j)に横置きを埋めるとき
    横が空いているとき(つまり(S>>j+1)&1==0のとき かつ j<W-1)
        dp[i][j+1][S|(1<<(j+1))] +=dp[i][j][S] ∵j+1 bitは必ず埋まる
    横が空いていないとき
        なにもしない(挿入できないので)

'''

# 入力
n = 3  # 行数
m = 3  # 列数

color = [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]]
dp = [[[0] * (1 << m) for _ in range(m)] for _ in range(n + 1)]
dp[0][0][0] = 1  # 0,0まで埋まっているときS==0の状態のみ存在する

for i in range(n):
    for j in range(m):
        for S in range(0, 1 << m):
            if color[i][j] or (S >> j) & 1:  # おけないとき
                if j < m - 1:
                    dp[i][j + 1][S & ~(1 << j)] += dp[i][j][S]
                else:
                    dp[i + 1][0][S & ~(1 << j)] += dp[i][j][S]
                continue  # おけないので終了

            # 縦におくとき
            if j == m - 1:  # 改行
                dp[i + 1][0][S | (1 << j)] += dp[i][j][S]
            else:
                dp[i][j + 1][S | (1 << j)] += dp[i][j][S]

            # 横に置くとき
            if (S >> (j + 1)) & 1 == 0 and j < m - 1:
                dp[i][j + 1][S | (1 << (j + 1))] += dp[i][j][S]

# print(*dp, sep='\n')
print(dp[n - 1][m - 1][0])  # 最後の端マスから見て,境界の状態がすべて0であればピッタリ埋まっているということ
```


#### P180 フィボナッチ数列
なぜかここから解説が嘘みたいにわかりやすくなる(DPの定義や更新の定式化をちゃんと書いてくれてる)。

```python
# 繰り返し二乗法の応用

MOD = 10 ** 4
import numpy as np

# 入力
n = int(input())

def matrix_pow(mat: np.matrix, n: int, mod: int):
    # nの2進数表記でビットが立っているところだけ処理すればいい
    mattmp = mat.copy()
    ret = np.matrix(np.eye(2, dtype='int32'))  # 単位元は対角行列
    while n > 0:
        if n & 1:  # ビットが立っているなら処理する
            ret *= mattmp
            ret %= mod
        mattmp = mattmp**2
        n >>= 1  # ビットを処理
    return ret

A = np.matrix([[1, 1],
               [1, 0]])

A_n = matrix_pow(A, n, MOD)
F1F0 = np.array([1, 0]).reshape(2, 1)
Fn = A_n * F1F0 % MOD

print(Fn[1, 0])
```

#### P182 Blocks
DPの定式化を行列形式に書き換えると、これも繰り返し二乗法でオーダーが落ちるっていうアイデア。
```python
import numpy as np
MOD = 10**4 + 7
def matrix_pow(mat: np.matrix, n: int, mod: int):
    # nの2進数表記でビットが立っているところだけ処理すればいい
    mattmp = mat.copy()
    ret = np.matrix(np.eye(3, dtype='int64'))  # 単位元は対角行列
    while n > 0:
        if n & 1:  # ビットが立っているなら処理する
            ret *= mattmp
            ret %= mod
        mattmp = mattmp**2
        n >>= 1  # ビットを処理
    return ret

# 入力
N = int(input())
A = np.matrix([[2, 1, 0],
               [2, 2, 2],
               [0, 1, 2]])
A_n = matrix_pow(A, N, mod=MOD)
print(A_n[0, 0])
```

#### P183 グラフの長さkのパスの総数
隣接行列の演算の意味を理解できる教育的な問題

```python
# この問題の解説もこれで終わり https://www.slideshare.net/hcpc_hokudai/advanced-dp-2016

import numpy as np

MOD = 10**4 + 7


def matrix_pow(mat: np.matrix, n: int, mod: int):
    mattmp = mat.copy()
    ret = np.matrix(np.eye(mat.shape[0], dtype='int64'))  # 単位元は対角行列
    while n > 0:
        if n & 1:  # ビットが立っているなら処理する
            ret *= mattmp
            ret %= mod
        mattmp = mattmp**2
        n >>= 1  # ビットを処理
    return ret


# 入力
n = 4
k = 2

G = np.matrix([[0, 1, 1, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1],
               [1, 0, 0, 0]])

ans = matrix_pow(G, 2, mod=MOD).sum()
print(ans)
```

#### P184 Matrix Power Series
```python
# 累乗の累積和を行列形式で書くとオーダー落とせるというアイデア
import numpy as np
def matrix_pow(mat: np.matrix, n: int, mod: int):
    mattmp = mat.copy()
    ret = np.matrix(np.eye(mat.shape[0], dtype='int64'))  # 単位元は対角行列
    while n > 0:
        if n & 1:  # ビットが立っているなら処理する
            ret *= mattmp
            ret %= mod
        mattmp = mattmp**2
        n >>= 1  # ビットを処理
    return ret

# 入力
n = 2
k = 2
M = 4
A = np.matrix([[0, 1],
               [1, 1]])

I = np.eye(A.shape[0])
AOII = np.hstack([A, np.zeros_like(A)])
AOII = np.vstack([AOII,
                  np.hstack([I, I])])

AOII_k = matrix_pow(AOII, k + 1, mod=M)

print((AOII_k[2:, :2] - I) % M)
```

#### P186 Minimizing maximizer
```python
# https://www.slideshare.net/hcpc_hokudai/advanced-dp-2016
# これもスライド図示がめっちゃわかりやすい
# 要はDPが区間クエリの処理を含むのでそこをセグ木で高速化できるという話

class SegmentTree:
    def __init__(self, ls: list, segfunc, identity_element):
        self.ide = identity_element
        self.func = segfunc
        self.n_origin = len(ls)
        self.num = 2 ** (self.n_origin - 1).bit_length()  # n以上の最小の2のべき乗
        self.tree = [self.ide] * (2 * self.num - 1)  # −1はぴったりに作るためだけど気にしないでいい
        for i, l in enumerate(ls):  # 木の葉に代入
            self.tree[i + self.num - 1] = l
        for i in range(self.num - 2, -1, -1):  # 子を束ねて親を更新
            self.tree[i] = segfunc(self.tree[2 * i + 1], self.tree[2 * i + 2])

    def __getitem__(self, idx):  # オリジナル要素にアクセスするためのもの
        if isinstance(idx, slice):
            start = idx.start if idx.start else 0
            stop = idx.stop if idx.stop else self.n_origin
            l = start + self.num - 1
            r = l + stop - start
            return self.tree[l:r:idx.step]
        elif isinstance(idx, int):
            i = idx + self.num - 1
            return self.tree[i]

    def update(self, i, x):
        '''i番目の要素をxに変更する(木の中間ノードも更新する) O(logN)'''
        i += self.num - 1
        self.tree[i] = x
        while i:  # 木を更新
            i = (i - 1) // 2
            self.tree[i] = self.func(self.tree[i * 2 + 1],
                                     self.tree[i * 2 + 2])

    def query(self, l, r):
        '''区間[l,r)に対するクエリをO(logN)で処理する。例えばその区間の最小値、最大値、gcdなど'''
        if r <= l:
            return ValueError('invalid index (l,rがありえないよ)')
        l += self.num
        r += self.num
        res_right = []
        res_left = []
        while l < r:  # 右から寄りながら結果を結合していくイメージ
            if l & 1:
                res_left.append(self.tree[l - 1])
                l += 1
            if r & 1:
                r -= 1
                res_right.append(self.tree[r - 1])
            l >>= 1
            r >>= 1
        res = self.ide
        # 左右の順序を保って結合
        for x in res_left:
            res = self.func(x, res)
        for x in reversed(res_right):
            res = self.func(res, x)
        return res


# 入力
n = 40
m = 6
s = [20, 1, 10, 20, 15, 30]
t = [30, 10, 20, 30, 25, 40]

# 0basedindexに
# tは半開区間のためそのまま
s = [ss - 1 for ss in s]
INF = 10**6
dp = SegmentTree([0] + [INF] * (n - 1), min,
                 identity_element=INF)  # DP配列をセグ木に乗っける(初期化済み)

for ss, tt in zip(s, t):
    mi = dp.query(ss, tt)
    dp.update(tt - 1, mi + 1)

print(dp[n - 1])  # ok
```

### つづく
時間がかかるかもしれないが次は中級編の後半をやっていこうと思う。

螺旋本をpythonで解いたシリーズ
(こっちのほうは比較的図示なども行っている。またこれから競技プログラミングを始めるならばこちらのほうがおすすめ。)

https://aotamasaki.hatenablog.com/entry/2019/10/11/%E8%9E%BA%E6%97%8B%E6%9C%AC%E3%82%92Python%E3%81%A7%E8%A7%A3%E3%81%8F_Part1

https://aotamasaki.hatenablog.com/entry/2019/11/03/%E8%9E%BA%E6%97%8B%E6%9C%AC%E3%82%92Python%E3%81%A7%E8%A7%A3%E3%81%8F_Part2

https://aotamasaki.hatenablog.com/entry/2019/12/01/%E8%9E%BA%E6%97%8B%E6%9C%AC%E3%82%92Python%E3%81%A7%E8%A7%A3%E3%81%8F_Part3

https://aotamasaki.hatenablog.com/entry/2019/12/11/%E8%9E%BA%E6%97%8B%E6%9C%AC%E3%82%92Python%E3%81%A7%E8%A7%A3%E3%81%8F_Part4