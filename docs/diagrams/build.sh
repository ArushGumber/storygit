#!/usr/bin/env bash
# Compile every diagram to PDF and SVG.
#
#   ./build.sh            # all diagrams
#   ./build.sh selection  # just docs/diagrams/selection.tex
#
# PDF is what presentable.tex includes; SVG is what the frontend's Architecture tab
# serves. Both are committed so the docs build without a TeX toolchain.
set -euo pipefail
cd "$(dirname "$0")"

targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
  mapfile -t targets < <(find . -maxdepth 1 -name '*.tex' ! -name 'style.tex' -printf '%f\n' | sed 's/\.tex$//' | sort)
fi

for name in "${targets[@]}"; do
  echo "==> $name"
  tectonic --keep-logs=false --print=false "${name}.tex"
  if command -v pdftocairo >/dev/null 2>&1; then
    pdftocairo -svg "${name}.pdf" "${name}.svg"
  elif command -v dvisvgm >/dev/null 2>&1; then
    dvisvgm --pdf "${name}.pdf" -o "${name}.svg"
  else
    echo "    no pdftocairo or dvisvgm on PATH; skipped SVG for $name" >&2
  fi
done
echo "done: ${#targets[@]} diagram(s)"
