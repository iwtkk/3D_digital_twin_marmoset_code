# なぜ高さ(垂直)方向は session ごとに補正するのか

## 結論

- **水平 (X, Y) は全 session 共通のグローバルスケーリング**を使っている。
- **垂直 (Z=高さ) だけ session ごとに affine 補正**している。
- 理由は「垂直方向の誤差量が session ごとに違う」ことが実データで分かったから。
  全 session で同じ値を使うと、多くの session で数十 mm の浮き/沈みが残る。

---

## そもそもベースは「全部一緒のスケーリング」

誤解しやすいが、土台は global スケーリングである。
`KP3D_CENTER` / `KP3D_NORM_SCALE`（測定空間の中心・半スパン）と
`BLENDER_CENTER` / `BLENDER_BBOX_SCALE`（Blender 側 bbox）は
**全 222 session をまとめて 1 セットだけ**算出した共通値。

per-session の affine `bZ_corrected = a*bZ + b` は、その global 変換の
**上に乗せる残差補正**であって、垂直軸だけに適用している。

つまり質問への直接の答えは：
**「実際にまず全部一緒のスケーリングをやった。その後に残る誤差が session ごとに違ったので、垂直だけ追加で個別補正した」**

---

## 全 session 共通だけでは不十分な理由（実データ）

per-session で求めた affine 係数の **session 間ばらつき**（全 222 session）:

| 係数 | 平均 | 標準偏差 | レンジ |
|---|---|---|---|
| slope `a`（スケール） | 0.971 | **0.032** | 0.905 〜 1.125 |
| intercept `b`（オフセット, m） | 0.003 | **0.034** | −0.122 〜 +0.086 |

もし全 session に共通の `(a, b)` を 1 つだけ適用したら、
この **session 間ばらつきがそのまま残差**になる：

- intercept のばらつき std ≈ **34 mm**
- slope のばらつき std 0.032 → bZ=1.5m（天井付近）で **約 47 mm** の高さ誤差

ユーザーの要求は「**数 mm でも浮いていると困る**」だったので、
30〜50 mm の残差は許容できない。
→ session ごとに `(a, b)` を出す必要があった。

---

## なぜ session ごとに誤差が変わるのか

kp3d は session ごとに別撮りのマルチカメラ三角測量で復元している。
session 単位でカメラ校正・基準の微妙なドリフトが入るため、
復元座標の **垂直方向のスケール・オフセットが session ごとに少しずつ違う**。

- これが「ある session は床ぴったり、別の session は数十 mm 浮く」という形で現れた。
- 水平方向（ケージ底面の広がり）は session 間で比較的安定していたため global で済ませた。
- 垂直方向は **床・足場・棒・天井という既知の物理高さ 4 点**でアンカーできるので、
  session ごとに高さ軸だけ直線フィット（affine）で合わせ込める。

---

## 「浮き」は誤差なのか行動なのか（重要な前提）

`scripts/dev/analysis_log.md` の分析で判明したこと：

- 「浮いて見える」session でも、個体が最も床に近づいた瞬間 (`floor_y_p99`) は
  床レベル付近（≈83 mm）にあった。
- median が高いのは、1 時間の大半を**棚・パーチの上で過ごす実際の行動**のため。

→ つまり「浮き」には 2 種類ある：
1. **行動由来**（個体が本当に高い所にいる）… 補正してはいけない
2. **session 由来の系統誤差**（座標系のスケール/オフセットずれ）… 補正すべき

per-session affine は、KDE で検出した **複数のプラットフォーム高さ**を
物理高さに合わせることで「2 の系統誤差だけ」を補正する。
床に一度も来ない no_contact session（10 件）では床ピークが取れないので、
`offset_only` や `no_peaks` にフォールバックして過補正を防いでいる。

---

## トレードオフ（per-session の注意点）

- **利点**: session ごとの系統誤差を消せる。物理高さに正確に合う。
- **欠点/リスク**: 1 session 内の行動分布に依存するので、
  プラットフォームをほとんど使わない session では誤った補正になりうる。
  → `MAX_MATCH_DIST_M`（0.10m 超のマッチを除外）やフォールバックで抑制している。

全 session 共通の方が「安全で頑健」だが「精度が粗い」。
今回は精度要求（数 mm）が厳しいので per-session を採用した、という判断。

---

## 関連ファイル

| パス | 役割 |
|---|---|
| `scripts/dev/analysis_log.md` (§6 浮き問題の分析) | 浮き=行動 vs 誤差 の診断根拠 |
| `outputs/dev/affine_validation/affine_summary.csv` | session 別 (a, b) のばらつき実データ |
| `scripts/dev/affine_pkl_csv/pkl_to_corrected_csv.py` | per-session affine 補正の実装 |
| `scripts/dev/affine_pkl_csv/KDE_AFFINE_QA.md` | KDE パラメータ・affine 回帰の詳細 |
