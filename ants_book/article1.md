蟻本Python回答集 初級編 (~P128)
===

### はじめに
とうとうAtCoder水色になりました(過去問精進と夜活コンテストのおかげ)。さらなる高みを目指すべく、プログラミングコンテストチャレンジブック [第2版] [■](https://book.mynavi.jp/ec/products/detail/id=22672) (通称、蟻本)を解くことした。
せっかくなのでPythonでの解答をここに記録する。

Pythonで解答してる人のブログを漁っても初級編の途中(DPとか)で挫折してる人が多そうだったので誰かの助けになれたらと思う。

著作権保護のため本の内容をすべて公開するわけではないので、解説などは本を見てほしい。
一方、変数名の説明なしにコード例が書いてある問題もいくつかあり、そういう問題はコードのコメントに意味合いを補足した。

この本の購入に関してはmynavi booksからPDFを購入するのがおすすめだ。ノートアプリを使っていろいろ書き込むことができる。

### 章 1-6 気軽にウォーミングアップ
#### P023 Ants
```python
# 入力例
L = 10
n = 3
x = [2, 6, 7]

# 最小の時間も最大の時間も一番端のもの
ans_min = min(L - x[-1], x[0])
ans_max = max(x[-1], L - x[0])
print(ans_min, ans_max)
```

#### P025 ハードルの上がった「くじびき」
P8の問題の解答も兼ねている。

```python
# P8の問題をN<1000にした場合
# O(N^2 log N)
from random import randint
from bisect import bisect_left

n = 1000
m = 10 ** 5  # ちょうどなってほしい数(ここでは決め打ち)
k = [randint(1, 2 * 10 ** 6) for _ in range(n)]

# 事前に、2つ選んだときにいくつになるか列挙しておく
candi = set()
for i in range(n):
    for j in range(i, n):
        candi.add(k[i] + k[j])
candi = sorted(candi)

# a+b、c+dの組み合わせで足すと=mになるのを二分探索で探す
for ab in candi:
    x = m - ab  # イコールになっててほしいやつ
    i = bisect_left(candi, x)
    if i != len(candi) and candi[i] == x:
        print('Yes')
        break
else:
    print('No')
```


### 章 2-1 すべての基本"全探索"
#### P034 部分和問題
あとのDPで更に効率的なやり方が出てくるが今は全探索で解く。

```python
n = 4
a = [1, 2, 4, 7]
k = 13

import sys
sys.setrecursionlimit(1 << 25)

def dfs(i, s):  # 足し算の状態を下に伝播して、作れるかの状態を上に伝播する
    # 終了条件
    if i == n:
        return s == k  # 合計がkになってればok
    # 使わなかった場合の探索
    flg1 = dfs(i + 1, s)
    # 使う場合の探索
    s += a[i]
    flg2 = dfs(i + 1, s)
    return flg1 or flg2  # どっちかがTrueならば良い

print('Yes' if dfs(0, 0) else 'No')
```

#### P035 Lake Counting
```python
# http://poj.org/problem?id=2386
# union find つかったらめっちゃ簡単に解けそうな気がするやつ
# 練習のためdfsで
import sys
sys.setrecursionlimit(1 << 25)
from itertools import product

N = 10
M = 12
MAP = '''W........WW.
.WWW.....WWW
....WW...WW.
.........WW.
.........W..
..W......W..
.W.W.....WW.
W.W.W.....W.
.W.W......W.
..W.......W.'''

def map_as(m, replace={'W': 1, '.': 0}):
     # 入力をいい感じに0,1の二重リストにしてくれる関数
    m = m.split()
    ret = []
    for line in m:
        ret.append([replace[s] for s in line])
    return ret

MAP = map_as(MAP)

def dfs(i, j):  # 周囲を探索しながら.に置き換える,何も返さない
    # 終了条件はなくても勝手に止まる
    for di, dj in product([-1, 0, 1], repeat=2):
        ni, nj = i + di, j + dj
        if not (0 <= ni < N and 0 <= nj < M):
            continue
        if MAP[ni][nj] == 1:
            MAP[ni][nj] = 0
            dfs(ni, nj)

ans = 0
for i, j in product(range(N), range(M)):
    if MAP[i][j] == 0:
        continue
    else:
        ans += 1
        dfs(i, j)
print(ans)
```

#### P037 迷路の最短路
簡単な(割に書くのがだるい)ので省略する。
AtCoderに全く同じ問題があるのでこちらの回答例を参考にしてほしい。

https://atcoder.jp/contests/atc002/tasks/abc007_3

### 章 2-2 猪突猛進！"貪欲法"

#### P042 硬貨の問題
こちらも非常に簡単なので省略する。
コインの額面が現実と異なる場合ではコイン問題を解くDPを書かないと行けない。コインの枚数はすべて貪欲法で解けるわけではないことに注意。

#### P043 区間スケジューリング問題
最近のコンテストでも出て、知っているか知らないかで大きく差がついた問題になっていた。

https://atcoder.jp/contests/keyence2020/tasks/keyence2020_b

```python
from operator import itemgetter
n = 5
s = [1, 2, 4, 6, 8]
t = [3, 5, 7, 9, 10]

st = [(ss, tt) for ss, tt in zip(s, t)]
# 選べる仕事の中で一番早く終る仕事を選ぶ→終わる順にソートしておくと便利
st.sort(key=itemgetter(1))
pre_t = 0  # 前の終了時刻
ans = 0
for s, t in st:
    if s <= pre_t:
        continue  # 前の終了時より前なので始められない
    ans += 1
    pre_t = t
print(ans)
```

#### P045 Best Cow Line

```python
# http://poj.org/problem?id=3617
N = 6
S = 'ACDBCB'

# Sの先頭か最後をTの末尾に追加しろ言っているのだから小さい方の文字を追加すれば良い
# ただし1文字比較では同じ文字だったときにバグるので文字列を保持して小さい方を選択する必要がある。
from collections import deque  # pythonはlist likeなオブジェクトの大小比較を辞書順でやってくれる
T = []
S_for = deque(S)
S_rev = deque(reversed(S))

while S_for and S_rev:
    if S_for <= S_rev:
        T.append(S_for.popleft())
        S_rev.pop()
    else:
        T.append(S_rev.popleft())
        S_for.pop()
print(''.join(T))
```

#### P047 Saruman's Army
```python
# http://poj.org/problem?id=3069
from bisect import bisect_left, bisect_right
N = 6
R = 10
X = [1, 7, 15, 20, 30, 50]
# 端からギリギリRになるところに印をつけて更新していく
# whileで愚直にシミュレーションをする
i = 0
ans = 0
while i < N:
    x_left = X[i]
    # ここでは二分探索で印をつける点と、次の左端の点のidxを見つける
    x_right = X[bisect_right(X, x_left + R, lo=i) - 1] + R
    i = bisect_right(X, x_right, lo=i)
    ans += 1
print(ans)
```

#### P049 Fence Repair
P075で再登場するのでここでは省略する。

### 章 2-3 値を覚えて再利用”動的計画法”
#### P052 01ナップサック問題
ココらへんの説明は正直わかりにくい。定式化がごちゃごちゃ変わったり再帰だったりDPだったり説明に一貫性がない。
ただP55のcolumnにある定式化がナップサックDPではよく用いられていると思うので、それに合わせて実装する。

```python
# 再帰関数版は省略、P55にあるDPを行う
N = 4
W = [2, 1, 3, 2]
V = [3, 2, 4, 2]
W_max = 5

'''
dp[i][j] ... 達成可能な最大の価値。[0,i)までの品物を重さがj以下で選んだとき。
更新式
dp[i+1][j] = max(dp[i][j], dp[i][j-W[i]] + V[i]) #i番目を取るか取らないかで良い方を選択する。
初期条件
dp[0][:]=0 品物を選んでないなら価値はない
dp[:][0]=0 重さ0では品物を選べない→価値はない
'''
dp = [[0] * (W_max + 1) for _ in range(N + 1)]
for i in range(N):
    for j in range(W_max + 1):
        dp[i + 1][j] = max(dp[i + 1][j], dp[i][j])
        if j - W[i] >= 0:
            dp[i + 1][j] = max(dp[i + 1][j], dp[i][j - W[i]] + V[i])

print(dp[-1][-1])
```

#### P056 最長共通部分列問題
編集距離も同様の処理で計算できる。

```python
n = 4
m = 4
s = 'abcd'
t = 'becd'

'''
dp[i][j] ... s[:i]とt[:j]のLCS長
更新則
dp[i+1][j+1] = max(dp[i][j+1],dp[j+1][j], dp[i][j]+1) #3項目はs[i]==t[j]のときだけ
∵同じ文字だったら最長が1が増える。そうじゃなかったら最長の方を選ぶ
'''
dp = [[0] * (m + 1) for _ in range(n + 1)]

# # 初期化を明示的に書くと下のように成るけどすべての要素を0で初期化してるので今はいらない
# for i in range(n + 1):
#     dp[i][0] = 0
# for j in range(m + 1):
#     dp[0][j] = 0

for i in range(n):
    for j in range(m):
        dp[i + 1][j + 1] = max(dp[i][j + 1],
                               dp[i + 1][j],
                               (dp[i][j] + 1) if s[i] == t[j] else 0)
print(dp[-1][-1])
```

#### P058 個数制限なしナップサック問題
```python
n = 3
W = [3, 4, 2]
V = [4, 5, 3]
W_max = 7
'''
dp[i][j] ... 価値の総和の最大。[0,i)の中から選んで、重さがj以下であるときの。
更新
dp[i+1][j] = max(dp[i][j], dp[i+1][j-W[i]]+V[i])
'''
# こっちのほうがコイン問題の更新を行うよりもシンプル！(本質は同じ)
dp = [[0] * (W_max + 1) for _ in range(n + 1)]
for i in range(n):
    for j in range(W_max + 1):
        dp[i + 1][j] = max(dp[i + 1][j], dp[i][j])
        if j - W[i] < 0:
            continue
        dp[i + 1][j] = max(dp[i + 1][j], dp[i + 1][j - W[i]] + V[i])
print(dp[-1][-1])
```

ちなみにコイン問題のDPを用いてもこれは解ける(dp[j]...重さがピッタリjになる中で最大の価値 を作る)。でも蟻本じゃコイン問題について解説されてない。螺旋本とか見ていただきたい。

コイン問題のDPで解いたver
```python
# コイン問題の応用で解けるので解いてみる
n = 3
W = [3, 4, 2]
V = [4, 5, 3]
W_max = 7

'''
dp[j] ... ピッタリ重さがjになる中で最大の価値を記録する
更新
dp[j+W[i]] = max(dp[j+W[i]], dp[j]+V[i])
'''

dp = [-1] * (W_max + 1)  # -1は作れないことを意味する
dp[0] = 0  # 重さがピッタリ0になるときは価値が0
for j in range(W_max):
    if dp[j] == -1:  # これが作れないということはここから先ちょうどの遷移は不可能
        continue
    for w, v in zip(W, V):
        if j + w > W_max:
            continue
        dp[j + w] = max(dp[j + w], dp[j] + v)
print(max(dp))  # W_max以下で最大の価値が抽出できる。
```

#### P060 01ナップサック問題その2
重さだと配列の大きさがデカすぎてしまう→発想の逆転で配列のインデックスを今度は価値に対応させるというもの。

```python
n = 4
W = [2, 1, 3, 2]
V = [3, 2, 4, 2]
W_max = 5

'''
dp[i][j] ... 達成可能な重さの総和の最小。[0,i)で価値がjになるように選んだとき。
更新則
dp[i+1][j] = min(dp[i][j], dp[i][j-V[i]]+W[i])
初期値
dp[i][j]=INF (for any i,j)
dp[0][0]=0
'''
INF = float('inf')
V_sum = sum(V)

dp = [[INF] * (V_sum + 1) for _ in range(n + 1)]
dp[0][0] = 0

for i in range(n):
    for j in range(V_sum + 1):
        dp[i + 1][j] = min(dp[i + 1][j], dp[i][j])
        if j - V[i] < 0:
            continue
        dp[i + 1][j] = min(dp[i + 1][j], dp[i][j - V[i]] + W[i])

# dp[n][j]<=Wを満たす最大のj
for j in range(V_sum, -1, -1):
    if dp[n][j] <= W_max:
        print(j)
        break
```

#### P062 個数制限付き部分和問題
個数制限なしナップサック問題と考え方が少し似てる。
部分和問題であるが、ちょうどjを作れるか否かではなく、ちょうどjを作れるときに個数をいくつ余らせることができるか という発想がトリッキー。

```python
n = 3
a = [3, 5, 8]
m = [3, 2, 2]
K = 17

'''
dp[i][j] ... ちょうどjをつくるときの、a[i]の余りの最大個数。(作れないときは-1)
更新則
dp[i+1][j] = m[i] for (dp[i][j]>=0)
dp[i+1][j] = dp[i+1][j-a[i]] - 1 for (上記でない かつ dp[i+1][j-a[i]] > 0)
dp[i+1][j] = -1 for (dp[i][j]<0 and dp[i][j-a[i]] < 0)
初期条件
dp[0][0] = 0 #これで十分駆動するはず
'''

dp = [[-1] * (K + 1) for _ in range(n + 1)]
dp[0][0] = 0
for i in range(n):
    for j in range(K + 1):
        if dp[i][j] >= 0:
            dp[i + 1][j] = m[i]
        else:
            if j - a[i] < 0:
                continue
            if dp[i + 1][j - a[i]] > 0:
                dp[i + 1][j] = dp[i + 1][j - a[i]] - 1
print('Yes' if dp[n][K] >= 0 else 'No')
```

#### P063 最長増加部分列問題
```python
from bisect import bisect_left, bisect_right

# 真に増加する部分列の最長を知りたい
n = 5
A = [4, 2, 3, 1, 5]

# 蟻本とは異なり、長さを可変にしておく(省メモリだしlen(dp)をするだけでLISが取得可能)
dp = []
for a in A:
    print(dp)
    idx = bisect_right(dp, a)  # 初めて真に大きい要素になるidx
    if idx == len(dp):
        dp.append(a)
    else:
        dp[idx] = a  # aに更新
print(dp)
print(len(dp))
```

#### P066 分割数
まず問題文がむずい。自分は「m個以下に分割」を「各グループに含まれる品物の個数がm個以下」なのかと勘違いして無限時間潰れた。
「mグループ以下に分割」というのが正しい解釈っぽい。

```python
n = 4
m = 3

'''
dp[i][j] ... jのi分割の総数
更新則
dp[i][j] = dp[i-1][j] + dp[i][j-i]
∵ jのi分割の定式化 →sum_{k=1..i} a_k = j
a_k=0がある時、これはi-1分割と定式化が同じになる→総数は等しい (=0は常にある)
a_k>0のとき、すべての要素(i個)から1を引くことができる。これはj-iのi分割の総数と等しくなる。
(ただしj-i>=0)
初期条件
dp[0][0]=1 #0の0分割は1通りと定義すると都合が良い(表を書き出してみるとわかりやすい)
'''

dp = [[0] * (n + 1) for _ in range(m + 1)]
dp[0][0] = 1

for i in range(1, m + 1):
    for j in range(n + 1):
        dp[i][j] = dp[i - 1][j] + (dp[i][j - i] if j - i >= 0 else 0)

print(dp[m][n])
print(*dp, sep='\n')
'''
[1, 0, 0, 0, 0]
[1, 1, 1, 1, 1]
[1, 1, 2, 2, 3]
[1, 1, 2, 3, 4]
この出力を見ればわかるように本来初期条件はdp[1][j]=1である。∵1分割はどんな数字でも必ず1通りになるから
しかし(0,0)に1を埋めておくと更新のときに全部それはやってくれるという仕組み
'''
```

#### P067 重複組合せ
個数制限付き重複組合せというべきか。高校で習ったような重複組合せ(nHm)は個数無制限であるので状況が違う。
今は、各商品について最大いくつ選べるかの制限がついている状況。その状況下での重複組合せ。
この問題は更新の高速化のためにしゃくとり法の考え方を用いていて難しい(初級編とは...)。

```python
n = 3
m = 3
a = [1, 2, 3]

'''
dp[i][j] ... [0,i)番目の品物からj個選ぶときの組み合わせの総数
更新則
dp[i+1][j] = sum_{k \in [max(0,j-a[i]),j+1) } dp[i][k]
∵i個目の商品を1個取るときの通り＋2個取るときの通り＋...
ただし尺取的に考えれば
dp[i+1][j] = dp[i+1][j-1] + dp[i][j] - dp[i][j-a[i]-1](ただし添字が正のときのみ)
初期条件
dp[0][0] = 1
'''

dp = [[0] * (m + 1) for _ in range(n + 1)]
dp[0][0] = 1
for i in range(n):
    for j in range(m + 1):
        dp[i + 1][j] = dp[i + 1][j - 1] + \
            dp[i][j] - \
            (dp[i][j - a[i] - 1] if j - a[i] - 1 >= 0 else 0)

print(dp[n][m])
print(*dp, sep='\n')
```

### 章 2-4 データを工夫して記録する"データ構造"
一番プログラミングっぽい章で好き。数学より、データ構造で高速化のほうがプログラミングって感じがしない？

#### P073 Expedition
通過したガソリンスタンドの権利を得て必要になったら権利を使うという発想が面白い。
```python
# http://poj.org/problem?id=2431
from heapq import heapify, heappop, heappush, heappushpop
class PriorityQueue:
    def __init__(self, heap):
        '''
        heap ... list
        '''
        self.heap = heap
        heapify(self.heap)
    def push(self, item):
        heappush(self.heap, item)
    def pop(self):
        return heappop(self.heap)
    def pushpop(self, item):
        return heappushpop(self.heap, item)
    def __call__(self):
        return self.heap
    def __len__(self):
        return len(self.heap)


N = 4
L = 25
P = 10
A = [10, 14, 20, 21]  # 座標
B = [10, 5, 2, 4]  # 補給量

A.reverse()
B.reverse()

# すでに通過したガソリンスタンドを使う権利があると解釈する
pq = PriorityQueue([])
# 距離Lはたかだか10**6なので距離を1づつforで回しても間に合う
ans = 0
for x in range(1, L):  # Lの手前までに1リットルあれば良い
    P -= 1
    if A and A[-1] == x:
        pq.push(-B[-1])
        del B[-1]
        del A[-1]
    if P == 0:
        if pq:
            P -= pq.pop()
            ans += 1
        else:
            print(-1)
            exit()
print(ans)
```


#### P075 Fence Repair
1番目に小さいものと2番目に小さいものを効率よく管理するのに優先度付きキューを用いると楽という話(高速でもある)。

```python
# http://poj.org/problem?id=3253
from heapq import heapify, heappop, heappush, heappushpop
class PriorityQueue:
    def __init__(self, heap):
        '''
        heap ... list
        '''
        self.heap = heap
        heapify(self.heap)
    def push(self, item):
        heappush(self.heap, item)
    def pop(self):
        return heappop(self.heap)
    def pushpop(self, item):
        return heappushpop(self.heap, item)
    def __call__(self):
        return self.heap
    def __len__(self):
        return len(self.heap)

N = 3
L = [8, 5, 8]

# 短いものからgreedyにマージしていくのが最適
# 一番目に短いものと次に短いものを高速に(わかりやすく)取得するためにpriority queueを用いる

pq = PriorityQueue(L)
ans = 0
while len(pq) > 1:
    mi1 = pq.pop()
    mi2 = pq.pop()
    new = mi1 + mi2
    ans += new
    pq.push(new)
print(ans)
```

#### P085 食物連鎖
与えられた動物の属する種類を全通り仮定して、仮定した事象の集合を管理することで効率よく解こうというアイデア。頭が良すぎる。

```python
# http://poj.org/problem?id=1182
class UnionFind:
    def __init__(self, N):
        self.N = N  # ノード数
        # 親ノードをしめす。負は自身が親ということ。
        self.parent = [-1] * N  # idxが各ノードに対応。
    def root(self, A):
        # print(A)
        # ノード番号を受け取って一番上の親ノードの番号を帰す
        if (self.parent[A] < 0):
            return A
        self.parent[A] = self.root(self.parent[A])  # 経由したノードすべての親を上書き
        return self.parent[A]
    def size(self, A):
        # ノード番号を受け取って、そのノードが含まれている集合のサイズを返す。
        return -self.parent[self.root(A)]
    def unite(self, A, B):
        # ノード番号を2つ受け取って、そのノード同士をつなげる処理を行う。
        # 引数のノードを直接つなぐ代わりに、親同士を連結する処理にする。
        A = self.root(A)
        B = self.root(B)
        # すでにくっついている場合
        if (A == B):
            return False
        # 大きい方に小さい方をくっつけたほうが処理が軽いので大小比較
        if (self.size(A) < self.size(B)):
            A, B = B, A
        # くっつける
        self.parent[A] += self.parent[B]  # sizeの更新
        self.parent[B] = A  # self.rootが呼び出されればBにくっついてるノードもすべて親がAだと上書きされる
        return True
    def is_in_same(self, A, B):
        return self.root(A) == self.root(B)


N = 100
K = 7
TXY = [(1, 101, 1),
       (2, 1, 2),
       (2, 2, 3),
       (2, 3, 3),
       (1, 1, 3),
       (2, 3, 1),
       (1, 5, 5), ]

# 頭良すぎか？
# x-A,x-B,x-Cをx,x+N,x+2Nに対応させる
uf = UnionFind(3 * N)
ans = 0
for t, x, y in TXY:
    x -= 1
    y -= 1
    if t == 1:
        # 矛盾チェック
        if uf.is_in_same(x, y + N) or uf.is_in_same(x, y + 2 * N):
            # もしxとyが別のグループに属しているなら同じ種類ということはできない
            ans += 1
        else:
            uf.unite(x, y)
            uf.unite(x + N, y + N)
            uf.unite(x + 2 * N, y + 2 * N)
    else:
        if uf.is_in_same(x, y) or uf.is_in_same(x, y + 2 * N):
            # もしxとyが同じ種類 か 食べられる関係性が逆の場合は矛盾
            ans += 1
        else:
            uf.unite(x, y + N)
            uf.unite(x + N, y + 2 * N)
            uf.unite(x + 2 * N, y)
print(ans)
```

### 章 2-5 あれもこれも実は”グラフ”
#### P093 二部グラフ判定
pythonの再帰は遅いので本とは異なり幅優先探索で判定している。
```python
from collections import deque
def is_bipartite_graph(graph, N):
    '''隣接リスト形式の入力を仮定'''
    # 再帰をしたくない(pypyの再帰は遅い)のでここでは幅優先探索で書く
    color = [-1] * N  # -1は無色。0,1で色を塗り分ける
    que = deque([(0, 0)])  # (ノード、色)
    color[0] = 0
    while que:
        u, c = que.popleft()
        for nx in graph[u]:
            if color[nx] == -1:  # 無色には色を塗る
                color[nx] = 1 - c
                que.append((nx, 1 - c))
            else:  # すでに色があるものに関しては、矛盾してないこと確認
                if color[nx] == c:
                    return False
    return True

# 入力例1
graph = {
    0: [1, 2],
    1: [0, 2],
    2: [0, 1],
}
print('Yes' if is_bipartite_graph(graph, 3) else 'No')

# 入力例2
graph = {
    0: [1, 3],
    1: [0, 2],
    2: [1, 3],
    3: [0, 2],
}
print('Yes' if is_bipartite_graph(graph, 4) else 'No')
```

#### P102 Roadblocks
二番目の最短路の長さを求める。
s→tの2番目の最短距離は、s→u(tの隣接)の1番目の最短距離と2番目の最短距離にu→tの距離を足した中に候補があるので、それを探索すれば良い。

```python
# http://poj.org/problem?id=3255
from collections import defaultdict
from heapq import heapify, heappop, heappush, heappushpop
class PriorityQueue:
    def __init__(self, heap):
        '''
        heap ... list
        '''
        self.heap = heap
        heapify(self.heap)
    def push(self, item):
        heappush(self.heap, item)
    def pop(self):
        return heappop(self.heap)
    def pushpop(self, item):
        return heappushpop(self.heap, item)
    def __call__(self):
        return self.heap
    def __len__(self):
        return len(self.heap)

N = 4
R = 4
graph = defaultdict(lambda: [])
graph[0] = [(1, 100)]
graph[1] = [(0, 100), (2, 250), (3, 200)]
graph[2] = [(1, 250), (3, 100)]
graph[3] = [(1, 250), (2, 100)]

# 2番目の最短距離まで持つDijkstra法
# s→tの2番目の最短距離は、s→uの1番目の最短距離と2番目の最短距離にu→tの距離を足した中に候補がある
# 各点について1,2番目の最短距離を持ってdijkstraを行えばよい

pq = PriorityQueue([])
dist = [float('inf')] * N  # 1番目の最短距離
dist2 = [float('inf')] * N  # 2番目の最短距離

dist[0] = 0
pq.push((0, 0))  # (距離, ノード)  # 1番目の交差点からスタートする
while pq:
    d, v = pq.pop()
    if dist2[v] < d:  # 扱っている距離が知りたい距離より大きいならなにもしない
        continue
    for to, cost in graph[v]:
        d_to = d + cost
        if dist[to] > d_to:  # 最短距離更新
            dist[to], d_to = d_to, dist[to]
            pq.push((dist[to], to))

        # 2番目の距離更新(1番目よりは大きくて、2番目よりは小さいものが更新対象)
        if dist2[to] > d_to and dist[to] < d_to:
            dist2[to] = d_to
            pq.push((dist2[to], to))

print(dist[N - 1])
print(dist2[N - 1])
```


#### P103 Conscription
親密度を負のコストだと見立てれば、最小全域木の順番に徴兵するのが一番コストのかからない順番になる(始めるノードによらない)のがミソ。

```python
# http://poj.org/problem?id=3723
# scipyを使って楽に実装できる
from scipy.sparse.csgraph import minimum_spanning_tree  # この関数の引数は隣接行列
from scipy.sparse import csr_matrix


N = 5
M = 5
R = 8

XYD = [(4, 3, 6831),
       (1, 3, 4583),
       (0, 0, 6592),
       (0, 1, 3063),
       (3, 3, 4975),
       #    (1, 3, 2049), #二重辺の処理がめんどくさいので ここで消しておく
       (4, 2, 2104),
       (2, 2, 781), ]


edges = []
n_nodes = N + M  # 前N個のノードを男性用にする
row = []
col = []
cost = []
for x, y, d in XYD:
    row.append(x)
    col.append(N + y)
    cost.append(-d)
    # col.append(x) #有向成分が片方しかなくてもminimum_spanning_treeはよしなにやってくれる
    # row.append(N + y)
    # cost.append(-d)
adj_mat = csr_matrix((cost, (row, col)),
                     shape=(n_nodes, n_nodes), dtype='int64')

mst = minimum_spanning_tree(adj_mat)
print(mst.sum())
print(mst)
print(int(10000 * n_nodes + mst.sum()))
```

#### P104 Layout

蟻本の解説が何を言っているのか正直わからない。
けど直感的な理解として、i＜jにおいてi→jは正の重力、j←iは負の重力が働いてるとして、紐をピンと伸ばしたときの最短距離と問題を言い換えることができる。こうして考えると、蟻本P105の図も納得。

```python
# http://poj.org/problem?id=3169

def bellman_ford(edges, s, N):
    '''
    edges ... (cost,from,to)を各要素に持つリスト
    s...始点ノード
    N...頂点数

    return
    ----------
    D ... 各点までの最短距離
    P ... 最短経路木における親
    '''
    P = [None] * N
    inf = float('inf')
    D = [inf] * N
    D[s] = 0
    for n in range(N):  # N-1回で十分だけど、N回目にもアップデートがあったらnegative loopを検出できる
        update = False  # 早期終了用
        for c, ot, to in edges:
            if D[ot] != inf and D[to] > D[ot] + c:
                update = True
                D[to] = D[ot] + c
                P[to] = ot
        if not update:
            break  # 早期終了
        if n == len(edges) - 1:
            print(-1)  # 負の閉路が存在するということはそのように並ぶことはできないということ
            exit()
            raise ValueError('NegativeCycleError')
    return D, P

# i<jにおいてi→jは正の重力、j←iは負の重力が働いてるとして、紐をピンと伸ばしたときの最短距離
# と問題を言い換えれば、蟻本P105の図も納得

N = 4
ML = 2
MD = 1
AL = [1, 2]
BL = [3, 4]
DL = [10, 20]
AD = [2]
BD = [3]
DD = [3]

edges = []
for a, b, d in zip(AL, BL, DL):
    edges.append((d, a - 1, b - 1))
for a, b, d in zip(AD, BD, DD):
    edges.append((-d, b - 1, a - 1))

# 未接続のi,i+1に対して順番がひっくり返らないようにする(つまりd[i]+0<=d[i+1]は0離れたい)
for i in range(N - 1):
    edges.append((-0, i + 1, i))

D, P = bellman_ford(edges, 0, N)
print(D)
print(P)

if D[N - 1] == float('inf'):
    print(-2)
else:
    print(D[N - 1])
```


### 章 2-6 数学的な問題を解くコツ
#### P107 線分上の格子点の個数
```python
P1 = (1, 11)
P2 = (5, 3)

# dx,dyを同じ数で割ったときどちらも整数ならば、grid point上にある(ベクトルをイメージするとわかりやすい)
# prid pointの数は最小のベクトル(dx/gcd(dx,dy),dy/gcd(dx,dy))の個数-1なので、gcd(dx,dy)-1となる

from math import gcd
print(gcd(P2[0] - P1[0], P2[1] - P2[1]) - 1)
```

#### P108 双六
蟻本ではかなり説明が端折られてるのでけんちょんさんのブログを読もう！
https://qiita.com/drken/items/b97ff231e43bce50199a

```python
def extgcd(a, b):
    '''ax + by = gcd(a,b) を満たすgcd(a,b),x,yを返す'''
    if b == 0:
        return a, 1, 0
    g, x, y = extgcd(b, a % b)
    return g, y, x - a // b * y

a = 4
b = 11

g, x, y = extgcd(a, b)

ans = []
if x >= 0:ans.append(x)
else:ans.append(0)
if y >= 0:ans.append(y)
else:ans.append(0)
if x < 0:ans.append(-x)
else:ans.append(0)
if y < 0:ans.append(-y)
else:ans.append(0)
print(*ans)
```


#### P110 素数判定
急に簡単な問題がしばらく続く(著者の頭が良すぎて難しいものと簡単なものの区別がついてない説)
```python
def is_prime(x: int):
    # 高速素数判定
    if x == 1:
        return False
    if x % 2 == 0:  # 定数倍高速化
        return x == 2
    for i in range(3, int(x**0.5) + 1, 2):
        if x % i == 0:
            return False
    return True

print('Yes' if is_prime(53) else 'No')
print('Yes' if is_prime(295927) else 'No')
```

#### P111 素数の個数
```python

def ret_eratos(N: int):
    '''エラトステネスの篩'''
    is_prime = [True] * (N + 1)
    is_prime[0] = False  # 0と1は素数ではない
    is_prime[1] = False
    for i in range(2, int(N ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * 2, N + 1, i):  # iの倍数は素数でない
                is_prime[j] = False
    return is_prime

is_prime = ret_eratos(1000000)
print(sum(is_prime[:13]))
print(sum(is_prime))
```


#### P113 区間内の素数の個数
ポイントは2点

- b-a<10^6 なのでこの区間内での走査は行える
- またこの区間内の倍数は√bまでの素数を列挙すればok

```python
def ret_eratos(N: int):
    '''エラトステネスの篩'''
    is_prime = [True] * (N + 1)
    # 0と1は素数ではない
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(N ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * 2, N + 1, i):  # iの倍数は素数でない
                is_prime[j] = False
    return is_prime


def range_eratos(a, b):
    '''[a,b)内の素数の配列 is_prime[x-a]で取得すること'''
    root_b = int(b**0.5) + 1
    is_prime_small = [True] * (root_b + 1)  # [0,√b)の篩
    is_prime_small[0] = False  # 0,1は素数でない
    is_prime_small[1] = False

    is_prime = [True] * (b - a)  # [a,b)の篩
    if a == 0:  # コーナーケース用
        is_prime[0] = False
        is_prime[1] = False
    elif a == 1:
        is_prime[0] = False

    for i in range(2, root_b + 1):
        if is_prime_small[i]:
            # smallの更新
            for j in range(i * 2, root_b + 1, i):
                is_prime_small[j] = False
            # is_primeの更新
            s = ((a - 1) // i + 1) * i  # a以上のiの倍数で最小のもの
            for j in range(max(2 * i, s), b, i):  # sが素数の可能性もあるのでmaxを取る
                is_prime[j - a] = False
    return is_prime


print(sum(range_eratos(22, 37)))
print(sum(range_eratos(22801763489, 22801787297)))

#検証用のコード
def verify(a, b):
    is_prime1 = ret_eratos(b - 1)
    is_prime2 = range_eratos(a, b)
    for i in range(a, b):
        if is_prime1[i] != is_prime2[i - a]:
            print(i, is_prime1[i], is_prime2[i])
    return True


verify(1000, 1000000)  # ok
verify(0, 1000000)  # ok
verify(1, 1000000)  # ok
```

#### P114 Carmichael Numbers
```python
# まず素数じゃないことを確かめてから
# 素直にxを[2,n)まで変化させて成り立つことを確かれば良い
# オリジナルの問題ではもっと高速化させないとTLEになるっぽい(そもそもpowを使わない)けど今はpowの練習なので

def is_prime(x: int):
    # 高速素数判定
    if x == 1:
        return False
    if x % 2 == 0:  # 定数倍高速化
        return x == 2
    for i in range(3, int(x**0.5) + 1, 2):
        if x % i == 0:
            return False
    return True


def solve(n):
    if is_prime(n):
        print('No')
        return
    for x in range(2, n):
        if pow(x, n, n) != x:  # pythonのpowに繰り返し二乗法は入っている
            print('No')
            return
    print('Yes')


solve(17)
solve(561)
solve(4)
```

### 章 2-7 GCJの問題に挑戦してみよう(1)
リンクが切れてて解答のverifyができないのがつらいところ。
そして本の解説が淡泊なのもつらいところ。

#### P117 Minimum Scalar Product
```python
# 大小関係を逆にソートする
# 逆にソートされた状態から適当なi,jをひっくり返すと内積が大きくなることが示せる。よって最適な状態であると言える。

def solve(v1, v2):
    v1.sort()
    v2.sort(reverse=True)
    return sum([x * y for x, y in zip(v1, v2)])

# 入力例1
n = 3
v1 = [1, 3, -5]
v2 = [-2, 4, 1]
print(solve(v1, v2))

# 入力例2
n = 5
v1 = [1, 2, 3, 4, 5]
v2 = [1, 0, 1, 0, 1]
print(solve(v1, v2))
```

#### P119 Crazy Rows
適切に前処理をし、満たすべき条件を整理すると簡単になる問題。

```python
from typing import List

def idx_last_one(rows: List[str]):
    ret = []
    for s in rows:
        last = -1
        for i in range(len(s)):
            if s[i] == '1':
                last = i
        ret.append(last)
    return ret

def solve(N: int, idxs_last_one: List[int]):
    ans = 0
    # 愚直にswapする
    for i in range(N):
        # i行目については last_idx<=iであるもののうち一番近いものを使うのが最適
        for j in range(i, N):
            last_idx = idxs_last_one[j]
            if last_idx <= i:
                break
        ans += j - i  # swapする回数は計算可能
        del idxs_last_one[j]
        idxs_last_one.insert(i, last_idx)
    print(ans)

# 入力例1
N = 2
mat = ['10',
       '11', ]
solve(N, idx_last_one(mat))

# 入力例2
N = 3
mat = ['001',
       '100',
       '010', ]
solve(N, idx_last_one(mat))

# 入力例3
N = 4
mat = ['1110',
       '1100',
       '1100',
       '1000']
solve(N, idx_last_one(mat))
```

#### P121 Bride the Prisoners
これがもしかして区間DPってやつなのか？
とりあえず写経してもう一回解説を読んだら理解が進んだ(自分の場合は)。

```python
'''
dp[i][j] ... (A[i],A[j])の区間に含まれる囚人を開放するのに必要な金貨の最小枚数
更新則
dp[i][j] = min(dp[i][k]+dp[k][j]) + A[j] - A[i] - 2
for j-i(=w)を2から(Q+1)まで広げつつ更新する
∵ 区間(A[i],A[j])の最小枚数は、抜く囚人をkとすると、
= dp[i][k]+dp[k][j] (P122図の①と②)
+A[j]-A[i] - 2 (区間の人数(-2はiの位置の分とkの位置の分))
となる。この中で最小値を探すってわけ。

初期条件
dp[q][q+1]=0 for q=[0,Q+1) ∵この範囲の中に取り出すべき囚人はいない(取引もないので金貨も必要ない)
'''


def solve(P, Q, A):
    A = [0] + A + [P + 1]
    dp = [[float('inf')] * (Q + 2) for _ in range(Q + 1)]
    # 初期化
    for q in range(Q + 1):
        dp[q][q + 1] = 0  # 隣の開放する囚人との間は誰も開放しないので金貨は0枚

    for w in range(2, Q + 2):  # 区間を広げていく
        for i in range(Q):
            j = i + w
            if j > Q + 1:
                break  # idxが範囲外で終了
            for k in range(i + 1, j):
                dp[i][j] = min(dp[i][j],
                               dp[i][k] + dp[k][j] + A[j] - A[i] - 2)

    print(*dp, sep='\n')
    print(dp[0][Q + 1])


# 入力例1
P = 8
Q = 1
A = [3]
solve(P, Q, A)

# 入力例2
P = 20
Q = 3
A = [3, 6, 14]
solve(P, Q, A)
```

#### P123 Millionaire
蟻本にdpの解説がないので他ブログを参考にした

https://lvs7k.github.io/posts/2018/11/pccb-easy-7/


いわゆる確率DPと言われるやつかもしれない(？)。独立性と排反事象から次の状態を作れないか考えるのが良さそう(?)。
```python
from itertools import product

def solve(M, P, X):
    '''
    dp[r][g] ... 最善戦略におけるクリアできる確率。最初の所持金がグループgに属するとき、残りrラウンドある場合。
    更新則
    dp[r + 1][g] = max_j (P * dp[r][g + j] + (1 - P) * dp[r][g - j])
    ∵ 残りr+1ラウンドで任意の金額を掛けたときに、勝つか負けるかで所持金が変化するのでg + j,g - jから遷移が来ることがわかる。
    r+1ラウンドで勝って結果的にクリアできる確率はP * dp[r][g + j] で
    r+1ラウンドで負けて結果的にクリアできる確率は(1 - P) * dp[r][g - j]である。
    よってr+1ラウンド目で結果的にクリアできる確率はその2つの和になる。
    賭ける金額によって確率が変動するので、最善戦略になるようにjについてmaxを取る。
    初期条件
    dp[0][-1]=1 ∵ 最初から10^6以上であれば賞金を受け取れる。
    '''
    n = pow(2, M) + 1  # 所持金のグループ数
    dp = [[0.0] * (n) for _ in range(M + 1)]
    dp[0][-1] = 1.0

    for r, g in product(range(M), range(n)):
        jub = min(g, n - g - 1)
        for j in range(jub + 1):
            dp[r + 1][g] = max(dp[r + 1][g],
                               P * dp[r][g + j] + (1 - P) * dp[r][g - j])

    print(dp[M][X * (n - 1) // (10**6)])
    print(*dp, sep='\n')

# 入力例1
M = 1
P = 0.5
X = 500000
solve(M, P, X)

# 入力例(original)
M = 2
P = 0.5
X = 500000
solve(M, P, X)

# 入力例2
M = 3
P = 0.75
X = 600000
solve(M, P, X)
```

### つづく
時間がかかるかもしれないが次は中級編の前半をやっていこうと思う。

螺旋本をpythonで解いたシリーズ
(こっちのほうは比較的図示なども行っている。またこれから競技プログラミングを始めるならばこちらのほうがおすすめ。)

https://aotamasaki.hatenablog.com/entry/2019/10/11/%E8%9E%BA%E6%97%8B%E6%9C%AC%E3%82%92Python%E3%81%A7%E8%A7%A3%E3%81%8F_Part1

https://aotamasaki.hatenablog.com/entry/2019/11/03/%E8%9E%BA%E6%97%8B%E6%9C%AC%E3%82%92Python%E3%81%A7%E8%A7%A3%E3%81%8F_Part2

https://aotamasaki.hatenablog.com/entry/2019/12/01/%E8%9E%BA%E6%97%8B%E6%9C%AC%E3%82%92Python%E3%81%A7%E8%A7%A3%E3%81%8F_Part3

https://aotamasaki.hatenablog.com/entry/2019/12/11/%E8%9E%BA%E6%97%8B%E6%9C%AC%E3%82%92Python%E3%81%A7%E8%A7%A3%E3%81%8F_Part4