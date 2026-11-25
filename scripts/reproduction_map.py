"""Generate the browsable reconstruction and host map from the same catalogue used by locate."""

import re
from pathlib import PurePosixPath
from urllib.parse import quote


def chapter(text):
    return re.sub(r"[^\w -]", "", text.strip().lower()).replace(" ", "-")


def link(title, path, section=None):
    if path.startswith("python -m "):
        module = path.split()[2]
        path = module.replace(".", "/") + "/__main__.py"
    target = "../" + quote(path, safe="/")
    if section:
        target += "#" + quote(chapter(section))
    return f"[{title}]({target})"


def render(index):
    families, routes = index["families"], index["transfer_routes"]
    groups = index["groups"]
    members = [key for keys in groups.values() for key in keys]
    if sorted(members) != sorted(families):
        raise ValueError("Every reconstruction family must occur in exactly one documentation group")
    lines = [
        "# 资产系列与迁入代码对照",
        "",
        (
            f"本版登记 {index['asset_count']:,} 个资产编号、{len(families)} 个重建系列和 {len(routes)} 条迁入或展示路线。"
            "同一次采集的不同盖板状态分别占一个编号。表中只列每个系列的代表编号，按任意编号运行 "
            "`python3 gen2hole/gen2hole.py locate 编号` 可以查看它的配方与迁入入口。"
            "[从干净克隆到首次运行](16-reproduction-cookbook.md) · [输入包与版本](01-downloads.md)"
        ),
        "",
        (
            "从“操作步骤”进入对应系列的文档；“重建”是实际程序入口，“配方”列出原始数据或已登记的参数。"
            "从“迁入路线”可跳到本页下方对应的宿主、实现代码和验收方法。"
        ),
    ]
    for group, keys in groups.items():
        lines.extend(
            [
                "",
                "## " + group,
                "",
                "| 系列及操作步骤 | 编号数 | 示例编号 | 重建、输入 | 迁入路线 |",
                "|---|---:|---|---|---|",
            ]
        )
        for key in keys:
            row = families[key]
            inputs = [link("重建", row["launcher"]), link("配方", row["recipe"])]
            if "prepare" in row:
                inputs.append(link("输入准备", row["prepare"]))
            choices = "、".join(f"[{r}](#route-{r})" for r in row["transfer"])
            entries = [
                link(row["name"], row["guide"], row["section"]),
                str(len(row["assets"])),
                row["assets"][0],
                " · ".join(inputs),
                choices,
            ]
            lines.append("| " + " | ".join(entries) + " |")
    lines.extend(
        [
            "", "## 按宿主选迁入入口", "",
            "运行 `python3 gen2hole/gen2hole.py routes 鱼眼` 可按宿主查路线，"
            "再用 `python3 gen2hole/gen2hole.py search --route 路线名` 查对应的资产系列。"
            "同一资产可以接入多个宿主，各路线的编号数有重叠。",
            "", "| 路线 | 适用场景 | 已登记系列 / 编号 | 代码 | 数据、运行和验收 |", "|---|---|---:|---|---|",
        ]
    )
    for key, row in routes.items():
        program = link(PurePosixPath(row["entry"]).name, row["entry"])
        supported = [family for family in families.values() if key in family["transfer"]]
        coverage = f"{len(supported)} / {sum(len(family['assets']) for family in supported):,}"
        lines.append(
            f'| <a id="route-{key}"></a>{key} | {row["name"]} | {coverage} | {program} | {link(row["section"], row["guide"], row["section"])} |'
        )
    lines.extend(
        [
            "",
            (
                "更新目录后运行 `python3 scripts/check-reproduction-index.py --write-map`。"
                "常规检查运行同一命令并省略参数，检查编号、源文件、指南章节、迁入路线与本页是否一致。"
            ),
        ]
    )
    return "\n".join(lines) + "\n"
