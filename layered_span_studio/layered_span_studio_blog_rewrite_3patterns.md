# Layered Span Studio 紹介記事 3案（平易な書き方で全面書き直し）

---

## 1案目: 標準版

重なり合う span を付けたいので、Layered Span Studio を作った
---

### はじめに

文字列アノテーションをしていると、たまに既存の道具ではやりにくい場面がある。

たとえば、同じ箇所に複数のラベルを付けたいときだ。NER でも、少し込み入った設定になると普通に出てくる。ところが、この条件になるだけで使えるツールがかなり減る。

さらに実際の作業では、ラベル定義を横で見たいし、同じ表層に過去どう付けたかも見たい。既存データをあとから流し込みたいこともある。

このあたりをまとめて扱えるものが手元になかったので、作った。

[Layered Span Studio](https://github.com/masakiaota/Layered-Span-Studio) という、重なり合う span を前提にした文字列アノテーションツールである。

必要な人はそこまで多くないと思う。ただ、必要な人にはかなり便利なはずだ。

（写真: 全体画面のスクリーンショット。左に document 一覧、中央に本文、右に関連情報が並んでいるもの）

### 何がやりたかったか

このツールでやりたかったことは、だいたい次の4つである。

- 重なり合うラベルをそのまま扱う
- ラベル定義を見ながら付与する
- 似た表層への過去の付与例をその場で見る
- 既存データを import して、最初から使える状態にする

個別には珍しくない。

ただ、これが全部そろっていないと、実際の作業が地味に面倒になる。ラベル定義は別ファイル、過去例は別画面、既存データは別スクリプト、みたいな状態になりやすい。

その結果、判断のたびに視線が散るし、作業のテンポも落ちる。

なので、アノテーションの画面の近くに必要な情報を寄せることを優先して作った。

### できること

現時点でできることはこんな感じである。

- テキストの任意区間に対して、重なり合う複数ラベルを付けられる
- 右ペインでラベル定義や既存アノテーションを確認できる
- 選択中の表層テキストについて、別 document の既存例を見られる
- shortcut で操作できる
- `project` `labels` `documents` を含む JSON を import できる

README では、デモプロジェクトを import してすぐ触れるようになっている。自前データを入れたい場合は、別途 import 用の手順書も置いてある。

（写真: 同じ付近に複数のラベルが付いている状態がわかる拡大スクリーンショット）

### 画面構成

画面はかなり素朴である。

- 左: document 一覧
- 中央: 本文
- 右: ラベル定義、関連する既存アノテーション、同じ表層の例

中央で span を選んで、右で判断材料を見る。やりたいのはそれだけである。

派手な機能はないが、この形のほうが実作業では扱いやすいと考えた。

（写真: 右ペインのスクリーンショット。ラベル定義や既存例が読めるもの）

### すぐ試す方法

README にある quickstart をそのままやれば動く。

```bash
# backend
cd backend
export JWT_SECRET='dev-secret'
uv sync
uv run scripts/create_user.py demo_login_user demo_login_pass
uv run uvicorn layered_span_studio_backend.main:app --host 127.0.0.1 --port 8000 --reload
```

```bash
# frontend
cd ../frontend
npm install
npm run dev
```

ブラウザで `http://127.0.0.1:5173` を開いてログインし、`Import Project` から `docs/quickstart-demo-project.json` を読み込めばよい。

README に書いてある demo 用のユーザー名とパスワードも用意してあるので、まずはそこから触るのが早い。

（写真: Project List 画面で Import Project ボタンが見えているスクリーンショット）

### 自前データの import

自前データも JSON で import できる。

最小構成はかなり単純で、top-level に少なくとも次の3つがあればよい。

- `project`
- `labels`
- `documents`

`annotations` は各 document の中に入れる。

import 手順書にも書いたが、最初は annotation なしで import を通してしまうのがおすすめである。先に文書とラベルだけ入れて、画面上で動作確認するほうが安全だ。

最小イメージはこういう形になる。

```json
{
  "project": {
    "name": "医療文書NER",
    "description": "自前データの初期 import"
  },
  "labels": [
    {
      "name": "疾患名",
      "color": "#D94841",
      "description": "疾患や病名に付与する"
    }
  ],
  "documents": [
    {
      "document_name": "record_001",
      "text": "患者は糖尿病の既往がある。",
      "status": "pending",
      "created_at": "2026-03-01T00:00:00Z",
      "updated_at": "2026-03-01T00:00:00Z",
      "annotations": []
    }
  ]
}
```

既存 project への追記 import にも対応している。細かい話は手順書のほうにまとめた。

### どういう人向けか

向いているのは、たとえば次のようなケースである。

- overlapping span を扱いたい
- multi-label な NER をやりたい
- ラベル定義や過去例を見ながら付与したい
- 既存の annotation データをまとめて取り込みたい

逆に、単純な 1 span 1 label だけを軽く付けたいなら、もっと小さい道具でも十分だと思う。

### まだ開発中

README にも書いたが、現時点ではまだ開発中である。

ただ、少なくとも「重なり合う span を付ける」「右で定義や既存例を見る」「JSON でデータを入れる」という土台はもう触れる状態まで来ている。

同じような用途の人がいたら、試してもらえると嬉しい。

- リポジトリ: [masakiaota/Layered-Span-Studio](https://github.com/masakiaota/Layered-Span-Studio)
- 自前データ import 手順: [docs/import-your-data.md](https://github.com/masakiaota/Layered-Span-Studio/blob/main/docs/import-your-data.md)

Issue や PR も歓迎である。


---

## 2案目: 技術寄り

overlapping span 用のアノテーションツールを作った
---

### はじめに

文字列アノテーションのツールは色々あるが、overlapping span をちゃんと扱いたいと思うと、急に選択肢が減る。

少なくとも自分が欲しかったのは、ただ span を塗れるだけのツールではなかった。

- 同じ箇所に複数ラベルを付けたい
- ラベル定義を見ながら作業したい
- 同じ表層への過去の付与例をすぐ見たい
- 既存のデータを import したい

この条件をまとめて満たしたかったので、[Layered Span Studio](https://github.com/masakiaota/Layered-Span-Studio) を作った。

名前の通り、重なり合う span を前提にした文字列アノテーションツールである。

（写真: アノテーション画面の全体スクリーンショット）

### 既存ツールで困った点

困るのは、span を付ける操作そのものではない。

実際には、付ける前後で参照したい情報が多い。

たとえば、ある表現にラベルを付けるとき、次のような確認をしたくなる。

- このラベルの定義は何だったか
- 似た表現には過去どう付けたか
- いま選択した文字列は、他の document ではどう扱われているか

この確認が別画面や別ファイルに散っていると、それだけで作業しづらい。

なので Layered Span Studio では、中央で本文を見ながら、右で判断材料も見られるようにした。

### 画面の考え方

画面構成はシンプルである。

- 左に document 一覧
- 中央に本文
- 右に定義や既存例

中央で span を選択し、右を見て判断する。基本はこれだけだ。

この構成にした理由は単純で、アノテーション作業では画面遷移が少ないほうがよいからである。

（写真: 右ペインの拡大スクリーンショット）

### 主な機能

現時点の主な機能は以下の通り。

- 重なり合う複数ラベルを付与できる
- ラベル定義をその場で確認できる
- 同一ラベルの既存 annotation を見られる
- 同じ表層テキストの既存 annotation を別 document から引ける
- shortcut に対応している
- JSON import ができる

README の quickstart では、demo project を import して動かせるようにしてある。

### 使い始めるまで

起動手順は普通である。backend と frontend を立ち上げる。

```bash
# backend
cd backend
export JWT_SECRET='dev-secret'
uv sync
uv run scripts/create_user.py demo_login_user demo_login_pass
uv run uvicorn layered_span_studio_backend.main:app --host 127.0.0.1 --port 8000 --reload
```

```bash
# frontend
cd ../frontend
npm install
npm run dev
```

その後、`http://127.0.0.1:5173` を開いてログインし、`Import Project` から demo 用 JSON を読み込めばよい。

（写真: Import Project の操作がわかるスクリーンショット）

### 自前データを入れるとき

自前データを入れる場合も、形式はそこまで複雑ではない。

最低限必要なのは、`project` `labels` `documents` である。`annotations` は document ごとに持つ。

最初から全部を完璧に入れようとすると、変換で詰まりやすい。なので、まずは文書とラベルだけで import を通し、その後に annotation を足すほうが無難だ。

たとえば最小例はこうなる。

```json
{
  "project": {
    "name": "医療文書NER",
    "description": "自前データの初期 import"
  },
  "labels": [
    {
      "name": "疾患名",
      "color": "#D94841",
      "description": "疾患や病名に付与する"
    }
  ],
  "documents": [
    {
      "document_name": "record_001",
      "text": "患者は糖尿病の既往がある。",
      "status": "pending",
      "created_at": "2026-03-01T00:00:00Z",
      "updated_at": "2026-03-01T00:00:00Z",
      "annotations": []
    }
  ]
}
```

詳しくは import 手順書にまとめた。

### 想定している用途

これは万人向けのアノテーションツールではない。

想定しているのは、次のような用途である。

- overlapping span を含む NER
- multi-label の span annotation
- 既存データを活用しながらの annotation
- 将来的に半自動化や LLM 連携も見据えた運用

逆に、画像 annotation や音声 annotation の用途ではないし、単純なラベル付けだけなら別の軽いツールでも十分だと思う。

### おわりに

必要になったときに、この手のツールは意外と見つからない。

なので、自分で使うために作った。

同じような作業をしている人には役に立つかもしれない。興味があれば見てほしい。

- リポジトリ: [masakiaota/Layered-Span-Studio](https://github.com/masakiaota/Layered-Span-Studio)
- README: [README.md](https://github.com/masakiaota/Layered-Span-Studio/blob/main/README.md)
- import 手順: [docs/import-your-data.md](https://github.com/masakiaota/Layered-Span-Studio/blob/main/docs/import-your-data.md)

Issue や PR も歓迎である。


---

## 3案目: 短め

文字列アノテーションで overlapping span を扱いたくて、ツールを作った
---

### 作ったもの

[Layered Span Studio](https://github.com/masakiaota/Layered-Span-Studio) という文字列アノテーションツールを作った。

重なり合う span を前提にしていて、同じ箇所に複数ラベルを付けられる。README にある demo project を import すれば、すぐ動かせるようにしてある。

（写真: 画面全体のスクリーンショット）

### 何が欲しかったのか

欲しかったのは、だいたい次のようなものだ。

- overlapping span をそのまま扱える
- ラベル定義を見ながら作業できる
- 同じ表層の過去例を確認できる
- 自前データを import できる

実際のアノテーションでは、span を付けること自体より、判断材料を参照する回数のほうが多いことがある。

だから、中央で本文を見て、右で必要な情報も見られる形にした。

### 画面

画面構成はこうなっている。

- 左: document 一覧
- 中央: 本文
- 右: ラベル定義、既存 annotation、同じ表層の例

大げさな話ではなく、単にこの配置が便利だったというだけである。

（写真: 右ペインのスクリーンショット）

### できること

今できることを並べると、こんな感じである。

- span の付与
- overlapping label の付与
- shortcut 操作
- demo project の import
- 自前データの import

自前データについては `project` `labels` `documents` を持つ JSON を読ませる形にしている。`annotations` は document の中に入れる。

最小例はこうである。

```json
{
  "project": {
    "name": "sample"
  },
  "labels": [
    {
      "name": "疾患名",
      "color": "#D94841",
      "description": "疾患や病名に付与する"
    }
  ],
  "documents": [
    {
      "document_name": "record_001",
      "text": "患者は糖尿病の既往がある。",
      "status": "pending",
      "created_at": "2026-03-01T00:00:00Z",
      "updated_at": "2026-03-01T00:00:00Z",
      "annotations": []
    }
  ]
}
```

細かいところは手順書に書いた。

### 試し方

起動は backend と frontend をそれぞれ立ち上げればよい。

```bash
# backend
cd backend
export JWT_SECRET='dev-secret'
uv sync
uv run scripts/create_user.py demo_login_user demo_login_pass
uv run uvicorn layered_span_studio_backend.main:app --host 127.0.0.1 --port 8000 --reload
```

```bash
# frontend
cd ../frontend
npm install
npm run dev
```

その後 `http://127.0.0.1:5173` を開いて、demo project を import する。

（写真: demo project を import している場面のスクリーンショット）

### どういう人向けか

向いているのは、重なりを含む文字列 annotation をやりたい人である。

単純な 1 span 1 label だけなら、別のもっと軽い道具で足りることも多いと思う。

逆に、ラベル定義や過去例を見ながら付けたい人、自前データを最初から入れたい人には合うはずだ。

### まだ開発中

まだ開発中ではあるが、もう触れる状態にはなっている。

興味があればどうぞ。

- リポジトリ: [masakiaota/Layered-Span-Studio](https://github.com/masakiaota/Layered-Span-Studio)
- import 手順: [docs/import-your-data.md](https://github.com/masakiaota/Layered-Span-Studio/blob/main/docs/import-your-data.md)
