# Method Specification

## Source of truth

This document is the authoritative source for the method, inputs, outputs, and directory conventions.

If existing scripts conflict with this document, this document is authoritative.

## Important

python, shell scriptを作成するときは必ず
- input-dir
- output-dir
- session
- help
を受け付けるようにして 

## Intermediate files

{target}/runのスクリプトは中間ファイルを必ず作成する
{target}/plotのスクリプトは中間ファイルを使用してプロットする

## Development target

- Target name: `{target}`

## Directory convention

### Editable source directories

Algorithm and execution scripts must be placed under:

```text
scripts/dev/{target}/run/
```

Plotting scripts must be placed under:

```text
scripts/dev/{target}/plot/
```

Generated output directories

Plot outputs must be saved under:

```text
scripts/dev/{target}_plot/
```

Generated data files such as CSV, TSV, JSON, and intermediate result files must be saved under:

```text
scripts/dev/{target}_files/
```

## about to make a directory

scripts/dev/{target}/run/
scripts/dev/{target}/plot/
の下にはディレクトリを作らない
