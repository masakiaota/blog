# Layered Span Studio 紹介記事 書き直し3案

---

## 1案目: 標準版

### 重なり合う span を付けたいので、Layered Span Studio を作った

文字列アノテーションをしていると、同じ箇所に複数のラベルを付けたい場面がある。NER でも少し込み入った設定になると普通に出てくる話なのに、この条件だけで使えるツールがかなり減る。

さらに実際の作業では、ラベル定義を横に出しておきたいし、同じ表層に過去どうラベルを付けたかも確認したい。既存データを最初から読み込みたいこともある。こういった判断材料が別ファイルや別画面に散っていると、作業のたびに視線が飛んでテンポが落ちる。

まとめて扱えるものが手元になかったので、作った。[Layered Span Studio](https://github.com/masakiaota/Layered-Span-Studio) という、重なり合う span を前提にした文字列アノテーションツールである。

（写真: 全体画面のスクリーンショット。左に document 一覧、中央に本文、右に関連情報が並んでいるもの）

---

### 画面構成

画面は素朴で、左に document 一覧、中央に本文、右にラベル定義・既存アノテーション・同じ表層の例を並べている。中央で span を選んで、右で判断材料を見る。やりたいのはそれだけだ。

アノテーション作業では、span を付ける操作そのものより、付ける前後の確認の回数のほうが多い。「このラベルの定義は何か」「似た表現には過去どう付けたか」「この文字列は別 document ではどう扱われているか」——これらが同じ画面の近くにあるだけで、判断の速度が変わる。

（写真: 右ペインのスクリーンショット。ラベル定義や既存例が読めるもの）

---

### できること

- 任意区間に対して、重なり合う複数ラベルを付けられる
- ラベル定義と既存アノテーションを右ペインで確認できる
- 選択中の表層テキストについて、別 document の既存例を参照できる
- shortcut で操作できる
- `project` `labels` `documents` を含む JSON を import できる

（写真: 同じ箇所に複数ラベルが付いている状態の拡大スクリーンショット）

---

### すぐ試す方法

backend と frontend をそれぞれ立ち上げる。

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

`http://127.0.0.1:5173` を開いてログインし、`Import Project` から `docs/quickstart-demo-project.json` を読み込めばよい。README にデモ用のユーザー名とパスワードも書いてあるので、まずはそこから触るのが早い。

（写真: Project List 画面で Import Project ボタンが見えているスクリーンショット）

---

### 自前データの import

自前データも JSON で import できる。`project` `labels` `documents` の3つが top-level にあれば動く。`annotations` は各 document の中に入れる形だ。

最初は annotation なしで import を通してしまうのがおすすめで、文書とラベルだけ先に入れて画面上で確認するほうが安全だ。最小構成はこうなる。

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

細かいところは [import 手順書](https://github.com/masakiaota/Layered-Span-Studio/blob/main/docs/import-your-data.md) にまとめた。

---

### おわりに

必要な人はそこまで多くないと思う。ただ、overlapping span を扱いたい・ラベル定義や過去例を見ながら付けたい・既存データをまとめて取り込みたい、というケースには合うはずだ。

まだ開発中ではあるが、土台は触れる状態になっている。同じような作業をしている人がいたら、試してもらえると嬉しい。

- リポジトリ: [masakiaota/Layered-Span-Studio](https://github.com/masakiaota/Layered-Span-Studio)
- import 手順: [docs/import-your-data.md](https://github.com/masakiaota/Layered-Span-Studio/blob/main/docs/import-your-data.md)

Issue や PR も歓迎である。

---
---

## 2案目: 技術寄り

### overlapping span 用のアノテーションツールを作った

文字列アノテーションのツールは色々あるが、overlapping span をちゃんと扱いたいと思うと、急に選択肢が減る。さらに自分が欲しかったのは、span を塗れるだけのツールではなく、ラベル定義を見ながら・過去の付与例を引きながら・既存データをそのまま持ち込んで使えるものだった。

この条件をまとめて満たすものが見つからなかったので、[Layered Span Studio](https://github.com/masakiaota/Layered-Span-Studio) を作った。

（写真: アノテーション画面の全体スクリーンショット）

---

### 設計の考え方

アノテーション作業で時間を取られるのは、span を付ける操作よりも付ける前後の確認だ。「このラベルの定義は」「似た表現には過去どうラベルを付けたか」「いま選んだ文字列は他の document ではどう扱われているか」——こういった確認が別画面や別ファイルに散ると、判断のたびにコンテキストスイッチが発生する。

そこで Layered Span Studio は、中央で本文を見ながら右で判断材料を参照できる構成にした。左が document 一覧、中央が本文、右がラベル定義・既存アノテーション・同一表層の例だ。画面遷移を減らすことを優先した設計である。

（写真: 右ペインの拡大スクリーンショット）

---

### 主な機能

- 重なり合う複数ラベルの付与
- ラベル定義のその場での確認
- 同一ラベルの既存アノテーション参照
- 同じ表層テキストの既存例を別 document から引く
- shortcut 対応
- JSON import（demo project および自前データ）

---

### セットアップ

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

`http://127.0.0.1:5173` を開いてログインし、`Import Project` から demo 用 JSON を読み込めば動く。

（写真: Import Project の操作がわかるスクリーンショット）

---

### 自前データの import

形式はシンプルで、`project` `labels` `documents` を top-level に持つ JSON を渡す。`annotations` は document ごとに持たせる。最初は annotation なしで import し、文書とラベルの状態で動作確認してから足していくほうが安全だ。

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

詳細は [import 手順書](https://github.com/masakiaota/Layered-Span-Studio/blob/main/docs/import-your-data.md) を参照してほしい。

---

### 向いている用途・向いていない用途

overlapping span を含む NER、multi-label の span annotation、既存データを活用しながらのアノテーション——こういった用途に向けて作っている。逆に、単純な 1 span 1 label であれば別の軽いツールで十分だし、画像や音声のアノテーションには対応していない。

---

### おわりに

必要になったときに意外と見つからないツールだったので、自分で作った。同じような作業をしている人の役に立てば嬉しい。

- リポジトリ: [masakiaota/Layered-Span-Studio](https://github.com/masakiaota/Layered-Span-Studio)
- import 手順: [docs/import-your-data.md](https://github.com/masakiaota/Layered-Span-Studio/blob/main/docs/import-your-data.md)

Issue や PR も歓迎である。

---
---

## 3案目: 短め

### 文字列アノテーションで overlapping span を扱いたくて、ツールを作った

[Layered Span Studio](https://github.com/masakiaota/Layered-Span-Studio) という文字列アノテーションツールを作った。同じ箇所に複数ラベルを重ねて付けられるのが基本の前提で、demo project を import すればすぐ動かせるようにしてある。

（写真: 画面全体のスクリーンショット）

---

### なぜ作ったか

欲しかったのは、overlapping span を扱えること、ラベル定義を見ながら作業できること、同じ表層への過去の付与例をその場で確認できること、自前データを最初から流し込めること——この4つが揃ったツールだ。

アノテーション作業では、span を付ける操作より判断材料を参照する回数のほうが多い。だから、中央で本文を見ながら右で必要な情報を参照できる構成にした。判断材料が近くにあるだけで作業のテンポは変わる。

（写真: 右ペインのスクリーンショット）

---

### 起動と使い方

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

`http://127.0.0.1:5173` を開いて demo project を import すれば動く。

自前データは `project` `labels` `documents` を持つ JSON を渡す形で、`annotations` は document の中に入れる。最小例はこうだ。

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

細かいところは [import 手順書](https://github.com/masakiaota/Layered-Span-Studio/blob/main/docs/import-your-data.md) に書いた。

（写真: demo project を import している場面のスクリーンショット）

---

### まとめ

単純な 1 span 1 label だけなら別の軽い道具で足りる。ただ、重なりを含む annotation をしたい・ラベル定義や過去例を見ながら付けたい・自前データを最初から入れたい、という人には合うと思う。まだ開発中だが、もう触れる状態にはなっている。

- リポジトリ: [masakiaota/Layered-Span-Studio](https://github.com/masakiaota/Layered-Span-Studio)
- import 手順: [docs/import-your-data.md](https://github.com/masakiaota/Layered-Span-Studio/blob/main/docs/import-your-data.md)
