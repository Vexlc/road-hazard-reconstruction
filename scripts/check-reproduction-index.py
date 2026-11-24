"""Check the complete ID-to-reconstruction and transfer map in a source-only checkout."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(entry):
    if entry.startswith("python -m "):
        module = entry.split()[2].replace(".", "/")
        entry = module + "/__main__.py"
    path = ROOT / entry
    if not path.exists():
        raise ValueError(f"Missing source or recipe: {entry}")
    return path


def main():
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
    print(f"{len(owners)} IDs, {len(index['families'])} reconstruction families, {len(used)} transfer routes, 20 paired pits")


if __name__ == "__main__":
    main()
