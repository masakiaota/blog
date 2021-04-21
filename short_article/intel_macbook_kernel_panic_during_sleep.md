intel macbookでスリープ中に電源が切れる問題とその解決方法
---

### 概要

Intel CPUのMacbook proの電源を切った覚えがないのに、蓋を開けると電源の付く音が流れることが数回発生した。しかも決まって前回は異常終了したという警告。

そこで原因を調べ、暫定的な解決法をここにメモをする。日本語では情報が出なかったので誰かの助けになれば嬉しい。

(ちなみにIntel macの15,16インチモデルばかりに発生している様子だった。)

### エラーメッセージ
自分の環境では以下のようなエラーメッセージが発生していた。

`Sleep transition timed out after 180 seconds while creating hibernation file or while calling rootDomain's clients about upcoming rootDomain's state changes.` と、あるようにスリープ中に異常が起きているようだ。

```
panic(cpu 8 caller 0xffffff8019289d2a): Sleep transition timed out after 180 seconds while creating hibernation file or while calling rootDomain's clients about upcoming rootDomain's state changes. Thread 0x3c697.
Backtracing specified thread
Backtrace (CPU 8), Frame : Return Address
0xffffffc1cc96b848 : 0xffffff8018bf9795 
0xffffffc1cce8b860 : 0xffffff8018bf9476 
0xffffffc1cce8b930 : 0xffffff8018bc6bb8 
0xffffffc1cce8b9e0 : 0xffffff8018bce0ac 
0xffffffc1cce8ba90 : 0xffffff8018b947ec 
0xffffffc1cce8bb00 : 0xffffff8018b94028 
0xffffffc1cce8bb40 : 0xffffff8018b1c206 
0xffffffc1cce8bb60 : 0xffffff80191d17e7 
0xffffffc1cce8be20 : 0xffffff8019291f92 
0xffffffc1cce8be60 : 0xffffff80191fcb59 
0xffffffc1cce8be80 : 0xffffff8018aff725 
0xffffffc1cce8bef0 : 0xffffff8018b00634 
0xffffffc1cce8bfa0 : 0xffffff8018a5f13e 

Process name corresponding to current thread: kernel_task
Boot args: chunklist-security-epoch=0 -chunklist-no-rev2-dev chunklist-security-epoch=0 -chunklist-no-rev2-dev

Mac OS version:
20D91

Kernel version:
Darwin Kernel Version 20.3.0: Thu Jan 21 00:07:06 PST 2021; root:xnu-7195.81.3~1/RELEASE_X86_64
Kernel UUID: C86236B2-4976-3542-80CA-74A6B8B4BA03
KernelCache slide: 0x0000000018800000
KernelCache base:  0xffffff8018a00000
Kernel slide:      0x0000000018810000
Kernel text base:  0xffffff8018a10000
__HIB  text base: 0xffffff8018900000
System model name: MacBookPro16,4 (Mac-A61BADE1FDAD7B05)
System shutdown begun: NO
Hibernation exit count: 0

以下略
```


### 解決方法

ターミナルで以下のコマンドを入力する。

```
sudo pmset hibernatemode 0
```

外国にも同じようなバグに遭遇している方がたくさんいて議論がされていた。上記のコマンドも以下のリンクに示す先人の知恵である。
https://discussions.apple.com/thread/8567772?page=3
https://apple.stackexchange.com/questions/311072/macos-high-sierra-restarting-by-itself-during-sleep

ただし、上記のコマンドを実行するとMacの設定が書き換わる。具体的には、スリープ時にRAMのバックアップがストレージに書き込まれなくなる。つまりスリープしたままバッテリー切れを起こすと、作業中の内容は全て失われてしまうことになる。

ただ現状このRAMのバックアップ時にMacが異常終了しているため、全く恩恵が受けられない状況である。おとなしく上記の設定をしておくことをおすすめする。


元に戻したい場合は以下のコマンドを実行する

```
sudo pmset hibernatemode 3 #これがデフォルト設定
```


おしまい。



