#!/usr/bin/env bash
# End-to-end smoke: boot the server in production mode and drive the core loop over HTTP.
#
# Proves the three things a unit test cannot: that the built frontend is actually served,
# that the API answers on a real socket, and that propose -> accept -> the tree changing
# works through the whole stack rather than through a TestClient.
#
# Uses the mock provider, so it needs no network and no quota, and it exits non-zero on
# any failure so it can run in CI.
#
#   scripts/e2e_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${STORYGIT_PORT:-8123}"
DB="$(mktemp -d)/e2e.db"
LOG="$(mktemp)"
PY=".venv/bin/python"

fail() { echo "FAIL: $*" >&2; [ -f "$LOG" ] && tail -30 "$LOG" >&2; exit 1; }

if [ ! -f frontend/dist/index.html ]; then
  echo "==> building the frontend"
  (cd frontend && npm run build >/dev/null 2>&1) || fail "frontend build"
fi

echo "==> booting on :$PORT"
STORYGIT_DB="$DB" STORYGIT_PORT="$PORT" STORYGIT_E2E=1 \
  "$PY" scripts/e2e_server.py >"$LOG" 2>&1 &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then break; fi
  sleep 0.5
done
curl -fsS "http://127.0.0.1:$PORT/api/health" >/dev/null || fail "server did not come up"

echo "==> health"
curl -fsS "http://127.0.0.1:$PORT/api/health" | grep -q '"ok":true' || fail "health"
curl -fsS "http://127.0.0.1:$PORT/api/health" | grep -q '"openrouter_enabled":false' \
  || fail "the metered provider must be locked"

echo "==> the built frontend is served"
curl -fsS "http://127.0.0.1:$PORT/" | grep -q '<div id="root">' || fail "index.html not served"
curl -fsS "http://127.0.0.1:$PORT/gallery" | grep -q '<div id="root">' \
  || fail "client-side routes must fall through to the app shell"

echo "==> tree"
TREE=$(curl -fsS "http://127.0.0.1:$PORT/api/tree")
echo "$TREE" | grep -q '"node_type":"story"' || fail "no story root"
BEFORE=$(echo "$TREE" | "$PY" -c "import json,sys; print(len(json.load(sys.stdin)['nodes']))")

node_of() {
  curl -fsS "http://127.0.0.1:$PORT/api/tree" | "$PY" -c "
import json, sys
nodes = [n for n in json.load(sys.stdin)['nodes'] if n['node_type'] == sys.argv[1]]
print(nodes[-1]['id'] if nodes else '')
" "$1"
}

# Walk the progressive levels the way a writer does: episode, then scene, then beat, then
# the prose for it. Each step proves the loop end to end at one more level.
for level in episode scene beat prose; do
  case "$level" in
    episode) parent=$(node_of story) ;;
    scene)   parent=$(node_of episode) ;;
    beat)    parent=$(node_of scene) ;;
    prose)   parent=$(node_of beat) ;;
  esac
  [ -n "$parent" ] || fail "no parent for $level"

  echo "==> propose $level"
  PROPOSED=$(curl -fsS -X POST "http://127.0.0.1:$PORT/api/propose" \
    -H 'content-type: application/json' \
    -d "{\"node_id\":\"$parent\",\"level\":\"$level\",\"intent\":\"go on\"}")
  PID=$("$PY" -c "
import json, sys
body = json.loads(sys.argv[1])
assert body['candidates'], 'no candidates came back'
first = body['candidates'][0]
assert first['axis_label'], 'a candidate reached the writer with no axis label'
assert first['delta_summary'], 'a candidate reached the writer with no delta summary'
print(first['proposal_id'])
" "$PROPOSED") || fail "propose $level"

  echo "==> accept $level"
  ACCEPTED=$(curl -fsS -X POST "http://127.0.0.1:$PORT/api/action/accept" \
    -H 'content-type: application/json' -d "{\"proposal_id\":\"$PID\"}")
  echo "$ACCEPTED" | grep -q '"snapshot_id"' || fail "accept $level returned no snapshot"
done

echo "==> the tree grew"
AFTER=$(curl -fsS "http://127.0.0.1:$PORT/api/tree" \
  | "$PY" -c "import json,sys; print(len(json.load(sys.stdin)['nodes']))")
[ "$AFTER" -gt "$BEFORE" ] || fail "the tree did not grow ($BEFORE -> $AFTER)"

echo "==> the world state was populated"
curl -fsS "http://127.0.0.1:$PORT/api/authorship" | grep -q '"sentences"' || fail "authorship"
curl -fsS "http://127.0.0.1:$PORT/api/ledger" | grep -q '"dial"' || fail "ledger"

echo "==> a bad request is a typed problem, not a traceback"
PROBLEM=$(curl -sS -X POST "http://127.0.0.1:$PORT/api/action/accept" \
  -H 'content-type: application/json' -d '{"proposal_id":"p_nope"}')
echo "$PROBLEM" | grep -qv "Traceback" || fail "a traceback leaked to the client"

echo "==> a rule the writer adds is a rule the writer can delete"
curl -sS -X POST "http://127.0.0.1:$PORT/api/ledger/style-note" \
  -H 'content-type: application/json' -d '{"text":"shorter sentences"}' >/dev/null
curl -sS "http://127.0.0.1:$PORT/api/ledger" | grep -q "shorter sentences" \
  || fail "the style note was not stored"
curl -sS -X POST "http://127.0.0.1:$PORT/api/ledger/style-note/remove" \
  -H 'content-type: application/json' -d '{"text":"shorter sentences"}' >/dev/null
curl -sS "http://127.0.0.1:$PORT/api/ledger" | grep -q "shorter sentences" \
  && fail "the style note survived being deleted"

echo
echo "PASS  $BEFORE -> $AFTER nodes, served from one process on :$PORT"
