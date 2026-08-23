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
# The Architecture tab serves the same SVGs the paper includes, so they are copied rather
# than regenerated -- one source, one look, no chance of the two drifting apart.
frontend_public="../../frontend/public/diagrams"
if [ -d "$(dirname "$frontend_public")" ]; then
  mkdir -p "$frontend_public"
  cp -f ./*.svg "$frontend_public/" 2>/dev/null || true
  echo "==> copied SVGs to frontend/public/diagrams"
fi

echo "done: ${#targets[@]} diagram(s)"
