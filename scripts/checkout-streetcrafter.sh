#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo="$root/work/street_crafter"
upstream=9937775ba12f069d363c61ad00ff977900660f9c

if [[ ! -e "$repo" ]]; then
    git clone --no-checkout "${STREETCRAFTER_SOURCE:-https://github.com/zju3dv/street_crafter.git}" "$repo"
    git -C "$repo" checkout --detach "$upstream"
fi
if [[ "$(git -C "$repo" rev-parse --show-toplevel)" != "$repo" || "$(git -C "$repo" rev-parse HEAD)" != "$upstream" ]]; then
    echo "Expected StreetCrafter $upstream at $repo" >&2
    exit 1
fi
patch="$root/streetcrafter/reconstruction.patch"
if git -C "$repo" apply --reverse --check "$patch" >/dev/null 2>&1; then
    echo "StreetCrafter patch already applied."
else
    git -C "$repo" apply "$patch"
fi
while IFS= read -r -d '' source; do
    destination="$repo/${source#"$root/streetcrafter/overlay/"}"
    if [[ -e "$destination" ]] && ! cmp -s "$source" "$destination"; then
        echo "Keeping existing local file: $destination" >&2
        continue
    fi
    install -D -m 644 "$source" "$destination"
done < <(find "$root/streetcrafter/overlay" -type f -print0)
