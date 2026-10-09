#!/usr/bin/env bash
# Write the third-party files every release archive carries.
#
#   scripts/ci/third_party.sh <features> <out_dir>   # e.g. audio,cuda dist-notices
#
# Writes three files to out_dir:
#   THIRD-PARTY-LICENSES.txt  every compiled-in crate's licence (cargo-about),
#                             plus the Llama 3.1 Community License section
#   LICENSE-LLAMA             the Llama 3.1 Community License, as splintr ships it
#   NOTICE                    the Llama attribution, then the notices for code
#                             boostr, voxr and voxr-narrate port from upstream
#
# Needs cargo-about and jq. Exits non-zero when a notice file is missing from
# a resolved crate, so an archive never ships without one.

set -euo pipefail

FEATURES="${1:?usage: third_party.sh <features> <out_dir>}"
OUT="${2:?usage: third_party.sh <features> <out_dir>}"
mkdir -p "$OUT"

META="$(cargo metadata --locked --features "$FEATURES" --format-version 1)"

# The directory of one resolved package, by name.
crate_dir() {
  local dir
  dir="$(jq -r --arg n "$1" \
    '[.packages[] | select(.name == $n) | .manifest_path] | first // empty' <<<"$META")"
  if [[ -z "$dir" ]]; then
    echo "third_party.sh: package $1 is not in the build for features $FEATURES" >&2
    exit 1
  fi
  dirname "$dir"
}

# One file inside a resolved package; refuses a missing one.
crate_file() {
  local path
  path="$(crate_dir "$1")/$2"
  if [[ ! -f "$path" ]]; then
    echo "third_party.sh: $1 ships no $2 (looked for $path)" >&2
    exit 1
  fi
  printf '%s\n' "$path"
}

LLAMA="$(crate_file splintr-vocab-llama3 LICENSE)"
cp "$LLAMA" "$OUT/LICENSE-LLAMA"

cargo about generate --locked --features "$FEATURES" about.hbs \
  -o "$OUT/THIRD-PARTY-LICENSES.txt"
{
  printf '\n================================================================================\n'
  printf 'Llama 3.1 Community License\n\nUsed by:\n'
  jq -r '.packages[] | select(.name == "splintr-vocab-llama3") | "- \(.name) \(.version)"' <<<"$META"
  printf '\n'
  cat "$LLAMA"
} >>"$OUT/THIRD-PARTY-LICENSES.txt"

{
  printf 'blazr is licensed under Apache-2.0 (see LICENSE).\n\n'
  printf 'Built with Llama.\n'
  printf 'Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.\n'
  printf 'blazr embeds the Llama 3 tokenizer vocabulary. LICENSE-LLAMA holds that licence.\n'
  printf 'THIRD-PARTY-LICENSES.txt holds the licence of every crate compiled into blazr.\n'
  for entry in boostr:THIRD_PARTY_NOTICES.md voxr:LICENSE-OTHERS voxr-narrate:LICENSE-OTHERS; do
    name="${entry%%:*}"
    file="${entry#*:}"
    printf '\n================================================================================\n'
    printf '%s: code ported from upstream projects (%s)\n\n' "$name" "$file"
    cat "$(crate_file "$name" "$file")"
  done
} >"$OUT/NOTICE"

ls -l "$OUT"
