"""Check the complete ID-to-reconstruction and transfer map in a source-only checkout."""
import argparse
import json
import re
from pathlib import Path

from urllib.parse import unquote, urlsplit

from reproduction_map import chapter, render

ROOT = Path(__file__).resolve().parents[1]


def source(entry):
    if entry.startswith("python -m "):
        module = entry.split()[2].replace(".", "/")
        entry = module + "/__main__.py"
    path = ROOT / entry
    if not path.exists():
        raise ValueError(f"Missing source or recipe: {entry}")
    return path


def check_navigation():
    """Follow relative links on source-checkout entry pages, including explicit HTML anchors."""
    pages = ["README.md", "docs/00-start-here.md", "docs/01-downloads.md",
             "docs/16-reproduction-cookbook.md", "docs/asset-code-index.md", "docs/23-road-debris.md"]
    anchors = {}
    count = 0
    for name in pages:
        page = ROOT / name
        for target in re.findall(r"\]\(([^)]+)\)", page.read_text(encoding="utf-8")):
            url = urlsplit(target)
            if url.scheme or url.netloc:
                continue
            dest = (page.parent / unquote(url.path)).resolve() if url.path else page
            if not dest.exists():
                raise ValueError(f"Missing local link in {name}: {target}")
            if url.fragment and dest.suffix == ".md":
                if dest not in anchors:
                    content = dest.read_text(encoding="utf-8")
                    headings = re.findall(r"^#+\s+(.+)$", content, re.M)
                    anchors[dest] = {chapter(h) for h in headings} | set(re.findall(r'<a id="([^"]+)"', content))
                if unquote(url.fragment) not in anchors[dest]:
                    raise ValueError(f"Missing section in {name}: {target}")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-map", action="store_true", help="Refresh docs/asset-code-index.md")
    args = parser.parse_args()
    index = json.loads((ROOT / "catalog/reconstruction-index.json").read_text())
    routes = index["transfer_routes"]
    owners, used = {}, set()
    for family, row in index["families"].items():
        for key in ("launcher", "recipe", "prepare"):
            if key in row:
                source(row[key])
        guide = source(row["guide"])
        headings = {line.lstrip("#").strip() for line in guide.read_text().splitlines() if line.startswith("#")}
        if row["section"] not in headings or not row["transfer"] or not row["assets"]:
            raise ValueError(f"Missing section, route or IDs: {family}")
        for route in row["transfer"]:
            source(routes[route]["entry"])
            source(routes[route]["guide"])
            used.add(route)
        for asset in row["assets"]:
            if asset.casefold() in owners:
                raise ValueError(f"Duplicate ID: {asset}")
            owners[asset.casefold()] = family
    if len(owners) != index["asset_count"] or used != set(routes):
        raise ValueError("Asset count or transfer-route coverage mismatch")
    kit = json.loads((ROOT / "catalog/common-pit-kit.json").read_text())["assets"]
    paired = {asset for row in index["families"].values() if "kitti-pair" in row["transfer"] for asset in row["assets"]}
    if len(paired) != 20 or paired != set(kit):
        raise ValueError("KITTI routing and 20-pit rebuild selection disagree")
    page = ROOT / "docs/asset-code-index.md"
    expected = render(index)
    if args.write_map:
        page.write_text(expected, encoding="utf-8")
    elif not page.is_file() or page.read_text(encoding="utf-8") != expected:
        raise ValueError("Refresh the asset code map with --write-map")
    links = check_navigation()
    print(f"{links} local navigation links checked")
    print(f"{len(owners)} IDs, {len(index['families'])} reconstruction families, {len(used)} transfer routes, 20 paired pits")


if __name__ == "__main__":
    main()
