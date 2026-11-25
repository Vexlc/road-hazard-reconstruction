"""Check the complete ID-to-reconstruction and transfer map in a source-only checkout."""
import argparse
import json
import re
import subprocess
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


def check_navigation(pages):
    """Follow relative links on source-checkout entry pages, including explicit HTML anchors."""
    anchors = {}
    count = 0
    for page in pages:
        name = page.relative_to(ROOT)
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



def check_release_links(pages):
    """Verify private GitHub release and attachment links without downloading the large assets."""
    repo = "Vexlc/road-hazard-reconstruction"
    prefix = f"https://github.com/{repo}/releases/"
    links = {}
    for page in pages:
        for url in re.findall(r"\]\(([^)]+)\)", page.read_text(encoding="utf-8")):
            if not url.startswith(prefix):
                continue
            parts = [unquote(p) for p in urlsplit(url).path.split("/")[4:]]
            if len(parts) == 2 and parts[0] == "tag":
                links.setdefault((parts[1], None), page)
            elif len(parts) == 3 and parts[0] == "download":
                links.setdefault((parts[1], parts[2]), page)
    releases = {}
    page_number = 1
    while True:
        endpoint = f"repos/{repo}/releases?per_page=100&page={page_number}"
        batch = json.loads(subprocess.check_output(["gh", "api", endpoint], text=True))
        for release in batch:
            if not release["draft"]:
                releases[release["tag_name"]] = {asset["name"] for asset in release["assets"] if asset["state"] == "uploaded"}
        if len(batch) < 100:
            break
        page_number += 1
    for (tag, name), page in links.items():
        if tag not in releases or (name is not None and name not in releases[tag]):
            raise ValueError(f"Missing published GitHub target in {page.relative_to(ROOT)}: {tag} / {name or ''}")
    print(f"{len({tag for tag, name in links})} linked releases and {sum(name is not None for tag, name in links)} distinct attachment URLs checked")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-map", action="store_true", help="Refresh docs/asset-code-index.md")
    parser.add_argument("--release-links", action="store_true", help="Also verify published URLs through an authenticated GitHub CLI")
    args = parser.parse_args()
    index = json.loads((ROOT / "catalog/reconstruction-index.json").read_text())
    routes = index["transfer_routes"]
    for key, route in routes.items():
        guide = source(route["guide"])
        headings = {line.lstrip("#").strip() for line in guide.read_text().splitlines() if line.startswith("#")}
        if route["section"] not in headings:
            raise ValueError(f"Missing transfer instructions for {key}: {route['section']}")
        source(route["entry"])
    grouped = [family for families in index["groups"].values() for family in families]
    if len(grouped) != len(set(grouped)) or set(grouped) != set(index["families"]):
        raise ValueError("Every reconstruction family must appear in exactly one navigation group")
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
            if route not in routes:
                raise ValueError(f"Unknown route {route} in {family}")
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
    pages = {ROOT / "README.md", ROOT / "gen2hole/README.md"}
    for directory in ("docs", "gen2hole/docs", "gen2hole/examples", "pipelines", "streetcrafter"):
        pages.update((ROOT / directory).rglob("*.md"))
    pages.update(ROOT / row["guide"] for row in (*index["families"].values(), *routes.values()))
    links = check_navigation(pages)
    if args.release_links:
        check_release_links(pages)
    print(f"{links} local navigation links checked")
    print(f"{len(owners)} IDs, {len(index['families'])} reconstruction families, {len(used)} transfer routes, 20 paired pits")


if __name__ == "__main__":
    main()
