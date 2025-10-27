#!/usr/bin/env bash
set -euo pipefail

# --- args ---
if [ $# -lt 1 ]; then
  echo "Usage: $0 <paths.json> [OUT_DIR]" >&2
  exit 1
fi
JSON="$1"
OUT_DIR="${2:-kubeconfigs}"
MERGED="${OUT_DIR}/merged.yaml"
mkdir -p "$OUT_DIR"

echo "📄 JSON file: $JSON"
echo "📂 Output directory: $OUT_DIR"
echo

# --- deps ---
for bin in k3d kubectl jq awk; do
  command -v "$bin" >/dev/null 2>&1 || { echo "❌ '$bin' not found in PATH" >&2; exit 1; }
done
if ! command -v yq >/dev/null 2>&1; then
  echo "ℹ️  'yq' not found: skipping 0.0.0.0 → 127.0.0.1 fix"
  YQ_AVAILABLE=0
else
  YQ_AVAILABLE=1
fi

echo "🔎 Reading list of k3d clusters..."
clusters_json="$(k3d cluster list -o json)"
mapfile -t CLUSTERS < <(echo "$clusters_json" | jq -r '.[].name')

if [ "${#CLUSTERS[@]}" -eq 0 ]; then
  echo "No k3d clusters found." >&2
  exit 1
fi

echo "📦 Exporting and merging kubeconfigs..."
KUBEFILES=()
for c in "${CLUSTERS[@]}"; do
  f="$OUT_DIR/${c}.yaml"
  k3d kubeconfig get "$c" > "$f"
  if [ "$YQ_AVAILABLE" -eq 1 ]; then
    yq -i '(.clusters[].cluster.server |= sub("0.0.0.0","127.0.0.1"))' "$f" || true
  fi
  KUBEFILES+=("$f")
done

export KUBECONFIG="$(IFS=: ; echo "${KUBEFILES[*]}")"
kubectl config view --merge --flatten > "$MERGED"
echo "✅ Merged kubeconfig created successfully into $MERGED"
echo

# --- Extract ordered unique clusters from JSON ---
# Stream all cluster names (skip PMU item at index 0),
# then dedupe in order using awk.
readarray -t ORDERED_CLUSTERS < <(
  jq -r '.paths[] | .[1:][]' "$JSON" | awk '!seen[$0]++'
)

if [ "${#ORDERED_CLUSTERS[@]}" -eq 0 ]; then
  echo "⚠️  No clusters found in JSON under 'paths'." >&2
  echo "👉 Example: {\"paths\":[[\"PMU-1\",\"cluster1\",...],[\"PMU-2\",\"cluster4\",...]]}"
  # Helpful debug:
  echo "🔎 Debug dump of .paths:"
  jq '.paths' "$JSON" || true
  exit 1
fi

echo "🗺️  Computed deploy order: ${ORDERED_CLUSTERS[*]}"
echo

# Normalize 'cluster1'/'cluster-1' → 'k3d-cluster-1'
normalize_to_ctx() {
  local raw="$1"
  raw="${raw//[[:space:]]/}"
  local norm
  norm="$(echo "$raw" | sed -E 's/^cluster-?([0-9]+)$/cluster-\1/i')"
  echo "k3d-${norm}"
}

echo "🧪 DEBUG plan (no actual deploy):"
echo "---------------------------------"
for c in "${ORDERED_CLUSTERS[@]}"; do
  ctx="$(normalize_to_ctx "$c")"
  echo "➡️  JSON cluster: '$c'  -> normalized context: '$ctx'"

  if kubectl --kubeconfig "$MERGED" config get-contexts -o name | grep -qx "$ctx"; then
    echo "   ✓ Context found."
  else
    echo "   ⚠️  Context '$ctx' NOT found in merged. Skipping."
    echo
    continue
  fi

  echo "   [DEBUG] Would run: kubectl --kubeconfig \"$MERGED\" --context \"$ctx\" apply -f openpdc.yaml"
  echo "   [DEBUG] Would run: kubectl --kubeconfig \"$MERGED\" --context \"$ctx\" get pods -n openpdc"
  echo
done

echo "🎯 Done (debug only)."
