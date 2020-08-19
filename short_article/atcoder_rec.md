AtCoderでPythonが再帰に弱い問題をどうにかしたい
===

### 概要
目次

{ここに目次挿入}

- 本記事の貢献
  - Python, Pypy, Cythonにおける再帰関数の速度比較
  - cythonをscript感覚で動かすコマンドの作成
  - (stackで書き換えるのが一番早かったというオチ)

### 背景
pythonで競技プログラミングしているみなさんこんにちは。
pythonは遅いので基本的にpypyで提出していることかと思う。
とはいえpypyも万能ではなく以下の理由でペナルティを食らうことがある。

- 文字列結合がおそすぎてTLE(これはリストに入れていって最後に''.joinすれば問題ない)
- DPテーブル作ったらMLE (numpyとnumbaで書き換えてpythonで提出しよう！)
- 再帰関数遅すぎてTLE (pythonで通るのを祈るしかない)

再帰関数が遅い問題に関する明確な解決法は今の所整理されてないように思える。
特に最近(言語アプデ後から)は制約も大きくなっていることから深い再帰でも高速に処理する方法が求められる。(O(NlogN)解法でN<=10^6が出たことも！)

ということで本記事では、再帰関数遅い問題に対処する解決法を提案したい。

### 解決法
結論から言うと以下の二点

- cythonで書き換える
- 余裕があったらstackで書き換える(pypy提出)

stackで書き換えるのは労力がいるのでcythonに書き換えるのが現実的だとおもう

stackで書き換える例は最後ちょっとだけ見せるとして、本記事では主に再帰関数を用いた実装で速度比較をする。


### 速度比較

こちらの問題を用いる。
https://atcoder.jp/contests/abc138/tasks/abc138_d

木構造上で累積和を作っていく問題。詳しくは解説を見ていただきたい。

速度はmax(各テストケースの実行時間)を用いることにする。

結果は以下のようになった。

| 言語                 | 実行時間 [ms] | メモリ [KB] |
| -------------------- | ------------: | ----------: |
| Python3 (3.4.3)      |          1995 |      248496 |
| Python (3.8.2)       |          1194 |      241604 |
| PyPy3(7.3.0)         |      2268以上 |     1369104 |
| **Cython (0.29.16)** |       **684** |   **66264** |

1行目のPython3は言語アップデート前の結果でTLEスレスレ。<br>
2行目によって言語のアップデートによって性能も向上していることが確認できる。<br>
3行目のPyPy3ではTLEした。もしいつもの癖でPyPy3に提出してしまうとペナルティを食らってしまう。<br>
4行目、大本命Cython。圧倒的に早くなおかつ省メモリ。多少定数倍が重い処理を実装してもTLEもMLEの心配もなさそう。

具体的なソースコードを以下に示す。飛ばしてもらってよい。

#### PythonとPypyの回答
```python
import sys
from collections import defaultdict
sys.setrecursionlimit(1 << 25)
read = sys.stdin.readline

def mina(*argv, sub=1): return list(map(lambda x: x - sub, argv))

def ints():
    return list(map(int, read().split()))

#読み込み
N, Q = ints()
tree = defaultdict(lambda: [])
for _ in range(N - 1):
    a, b = mina(*ints())
    tree[a].append(b)
    tree[b].append(a)
cnt = [0] * N
for _ in range(Q):
    q, x = ints()
    cnt[q - 1] += x
cnt.append(0)  # -1アクセス用

#本処理
def dfs(u, p):  # uは現在のノード、pは親のノード
    cnt[u] += cnt[p]
    for nv in tree[u]:
        if nv == p:
            continue
        dfs(nv, u)
dfs(0, -1)
print(*cnt[:-1])
```

#### Cythonの回答
```python
import sys
from collections import defaultdict
sys.setrecursionlimit(1 << 24)
read = sys.stdin.readline
ra = range
enu = enumerate

def mina(*argv, sub=1): return list(map(lambda x: x - sub, argv))

cdef ints():
    return list(map(int, read().split()))

# 宣言
cdef:
    long N, Q, a, b, q, x, _
    long cnt[200005]

# 読み込み
N, Q = ints()
tree = [[] for _ in range(N)]
for _ in range(N - 1):
    a, b = mina(*ints())
    tree[a].append(b)
    tree[b].append(a)

for _ in ra(Q):
    q, x = ints()
    cnt[q - 1] += x

# dfsでcntに木に沿った累積和をsetしていく
cdef dfs(int u, int p):  # uは現在のノード、pは親のノード
    cdef long nv
    cnt[u] += cnt[p]
    for nv in tree[u]:
        if nv == p:
            continue
        dfs(nv, u)

dfs(0, N + 1)
for i in range(N):
    print(cnt[i], end=' ')
```



### Cythonを使いやすく

以上の議論より、再帰が深い場合はCythonを用いれば良さそうだ。
しかし`python hoge.py`のようにdirectlyにCythonを実行できないのが難点である。
(コンパイルに手間でテストやデバッグがしにくいのがCython人口の少ない理由だろう)

Cythonを実行するためには、C言語に変換→バイナリに変換→python内で呼び出し をする必要がある。

これらを一括でやってくれる`run_cython`なるコマンドを自分で定義してやれば、cythonによる競技プログラミングがぐっと楽になるはずだ。

cythonがinstall済みであることを前提に、fishを用いた実装例を以下に示す(ほかのshellを使っている人は書き換えてね)

```sh
function run_cython
    set stem (string split ".pyx" "" $argv); and\
    cythonize -3 -i $argv > /dev/null ; and\
    python -c "import $stem"
end
```

`run_cython hoge.pyx`というふうにcythonをスクリプト感覚で動かすことができる。 (初回コンパイル時には少し時間がかかる)


### 今回のオチ
再帰関数を高速化しようという目的だったが、stackでゴリゴリに書き換えてpypyで提出したのが一番早かったというオチ。
コンテスト中にやるのは時間の無駄な気もするが、定数倍高速化が超きつくてコンテスト時間に余裕がある場合は選択肢の一つになるかもしれない(いや、ないか)。

| 言語                         | 実行時間 [ms] | メモリ [KB] |
| ---------------------------- | ------------: | ----------: |
| Python3 (3.4.3)              |          1995 |      248496 |
| Python (3.8.2)               |          1194 |      241604 |
| PyPy3(7.3.0)                 |      2268以上 |     1369104 |
| Cython (0.29.16)             |           684 |       66264 |
| **PyPy (7.3.0) (stack再帰)** |           558 |      169308 |

ちなみにstackを用いた実装を他の言語に投げても1000ms程度だった。

ソースコード

```python
import sys
from collections import defaultdict
read = sys.stdin.readline

def mina(*argv, sub=1): return list(map(lambda x: x - sub, argv))

def ints():
    return list(map(int, read().split()))

#入力
N, Q = ints()
tree = defaultdict(lambda: [])
for _ in range(N - 1):
    a, b = mina(*ints())
    tree[a].append(b)
    tree[b].append(a)
cnt = [0] * N
for _ in range(Q):
    q, x = ints()
    cnt[q - 1] += x
cnt.append(0)  # -1アクセス用

# dfsでcntに木に沿った累積和をsetしていく
def dfs(u, p):  # 戻り値なしver
    S_args = [(u, p)]  # 引数管理のstack
    S_cmd = [0]  # 0:into, 1:outofの処理をすべきと記録するstack
    def into(args):
        '''入るときの処理'''
        u, p = args
        cnt[u] += cnt[p]
    def nxt(args):
        S_args.append(args)  # 抜けるときに戻ってくることを予約
        S_cmd.append(1)
        '''今の引数からみて次の引数を列挙'''
        u, p = args
        for nx in tree[u]:
            if nx == p:
                continue
            _stack(nx, u)
    def outof(args):
        '''抜けるときの処理'''
        pass
    def _stack(*args):  # お好きな引数で
        S_args.append(args)
        S_cmd.append(0)
    while S_cmd:
        now_args = S_args.pop()
        cmd = S_cmd.pop()
        if cmd == 0:
            into(now_args)
            nxt(now_args)  # 次の再帰する(次のintoを予約)
        else:
            outof(now_args)
dfs(0, -1)
print(*cnt[:-1])
```


### まとめ

**目的** : AtCoderにおいてPythonの再帰解法を高速化したい <br>
**結論** : 型付Cython使え


| 言語                 | 実行時間 [ms] | メモリ [KB] |
| -------------------- | ------------: | ----------: |
| Python3 (3.4.3)      |          1995 |      248496 |
| Python (3.8.2)       |          1194 |      241604 |
| PyPy3(7.3.0)         |      2268以上 |     1369104 |
| **Cython (0.29.16)** |       **684** |   **66264** |



そしてCythonをpython感覚で実行するコマンドをおいておく
```sh
function run_cython
    set stem (string split ".pyx" "" $argv); and\
    cythonize -3 -i $argv > /dev/null ; and\
    python -c "import $stem"
end
```