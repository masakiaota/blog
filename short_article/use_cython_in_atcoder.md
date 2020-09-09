AtCoderでCythonの力を開放する魔術詠唱
===

### 概要
以下のformatを**Python**で提出すればいい

```python
mycode = r'''
# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
{ここにcythonのコードを書く}
'''

import sys
import os
if sys.argv[-1] == 'ONLINE_JUDGE':  # コンパイル時
    with open('mycode.pyx', 'w') as f:
        f.write(mycode)
    os.system('cythonize -i -3 -b mycode.pyx')

import mycode
```

### AtCoderにおけるCython提出の弱点
Cythonの真の力は、既存のCやC++のコードをラップして利用することができる点である。
たとえば、`from libcpp.vector cimport vector`や`from libcpp.map cimport map`によってC++のSTLコンテナを利用することができる。

しかしAtCoderでは、CythonはすべてCへ変換されてしまうため、C++の便利なデータ構造を用いることができない。
この現状のために、たとえば現段階では、高速に動的に配列を確保する手段(vector)がない(自分で書けばいいが...)。Cythonでlistを使っても内部ではPythonを呼び出しているために素のPythonとさほど変わらない実行時間になるだろう。これは例えばグラフを扱うような問題では不利である。


### 解決方法
Python提出の際にコンパイルフェーズがあることを利用する。

AtCoderでは、テストケースが入力される前に、引数が`sys.argv[-1] = 'ONLINE_JUDGE'`となるような実行が走る。この際に、Cythonのコードを自分の都合の良いように(つまりC++に変換)コンパイルして、同一directoryにおいてしまえば、Cythonを利用してテストケースを高速に実行できる。

それが冒頭にも示した下記のコードだ。mycodeの冒頭のshebangのようなものは、CythonをC++に変換し高速化するオプションを記載したものである。

```python
mycode = r'''
# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
{ここにcythonのコードを書く}
'''

import sys
import os
if sys.argv[-1] == 'ONLINE_JUDGE':  # コンパイル時
    with open('mycode.pyx', 'w') as f:
        f.write(mycode)
    os.system('cythonize -i -3 -b mycode.pyx')

import mycode
```

### 性能評価
CythonとPythonで問題を解いてどれぐらい高速化されるのか見てみる。

UnionFindを使うこの問題を説いてみる。

https://atcoder.jp/contests/abc177/tasks/abc177_d

結果は以下のようになった。

|        | 実行時間 [ms] |
| ------ | ------------: |
| Python |           629 |
| Cython |            77 |

Pythonの起動に22msほどかかることから、処理部分については、11倍もの高速化になっていることが読み取れる



#### Pythonの回答
```python
import sys
sys.setrecursionlimit(1 << 25)
readline = sys.stdin.buffer.readline
read = sys.stdin.readline  # 文字列読み込む時はこっち

def ints(): return list(map(int, readline().split()))

class UnionFind:
    def __init__(self, N):
        self.N = N  # ノード数
        self.n_groups = N  # グループ数
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
        self.n_groups -= 1

        return True

    def is_in_same(self, A, B):
        return self.root(A) == self.root(B)


N, M = ints()
uf = UnionFind(N)
for _ in range(M):
    a, b = ints()
    uf.unite(a-1, b-1)

print(-min(uf.parent))
```

#### Cythonの回答

```python
# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True
 
ctypedef long long LL
from libc.stdio cimport scanf
from libcpp.vector cimport vector
ctypedef vector[LL] vec
 
 
cdef class UnionFind:
    cdef:
        LL N,n_groups
        vec parent
 
    def __init__(self, LL N):
        self.N = N  # ノード数
        self.n_groups = N  # グループ数
        # 親ノードをしめす。負は自身が親ということ。
        self.parent = vec(N,-1)  # 長くならないのでvectorを使う必要はないがせっかくなので
 
    cdef LL root(self, LL A):
        # ノード番号を受け取って一番上の親ノードの番号を帰す
        if (self.parent[A] < 0):
            return A
        self.parent[A] = self.root(self.parent[A])  # 経由したノードすべての親を上書き
        return self.parent[A]
 
    cdef LL size(self, LL A):
        # ノード番号を受け取って、そのノードが含まれている集合のサイズを返す。
        return -self.parent[self.root(A)]
 
    cdef bint unite(self,LL A,LL B):
        # ノード番号を2つ受け取って、そのノード同士をつなげる処理
        A = self.root(A)
        B = self.root(B)
 
        # すでにくっついている場合
        if (A == B):
            return False
 
        # 大きい方に小さい方をくっつけたほうが処理が軽いので大小比較
        if (self.size(A) < self.size(B)):
            A, B = B, A
 
        self.parent[A] += self.parent[B]  # sizeの更新
        self.parent[B] = A 
        self.n_groups -= 1
 
        return True
 
    cdef bint is_in_same(self,LL A,LL B):
        return self.root(A) == self.root(B)
 
 
cdef LL N,M,_
scanf('%lld %lld',&N, &M)
 
cdef UnionFind uf = UnionFind(N)
cdef LL a,b
for _ in range(M):
    scanf('%lld %lld',&a, &b)
    uf.unite(a-1, b-1)
 
print(-min(uf.parent))
```

### まとめ
本記事では、AtCoderにおいてCythonの力を開放するコードを紹介した。

この方法を用いると、いままでは使えなかったC++のSTLコンテナがCythonでも利用可能となる。たとえば`vector`(Pythonのlist)や`map`(Pythonのdict)、`deque`などである。

使えるものの一覧はこの中にある。
https://github.com/cython/cython/tree/master/Cython/Includes/libcpp

これでCythonは書きやすさの恩恵、numpyなどのライブラリの恩恵、さらにC++の速度の恩恵を享受することになるので、これから人気が出てくるのではないかと思う。懸念すべきはサポートしているeditorが壊滅的な点と情報が少ない点だろう。

そして最後にCythonをpython感覚で実行するコマンドをおいておく。登録しておくと便利だとおもう。下記はfish shellのものであるが他のshellについても少し書き換えれば動くだろう。
```sh
function run_cython
    set stem (string split ".pyx" "" $argv); and\
    cythonize -3 -i $argv > /dev/null ; and\
    python -c "import $stem"
end
```

Cythonユーザーが一人でも増えることを期待して。

### 参考文献

https://atcoder.jp/contests/language-test-202001/submissions/9878658


https://nagiss.hateblo.jp/entry/2020/09/08/203701



