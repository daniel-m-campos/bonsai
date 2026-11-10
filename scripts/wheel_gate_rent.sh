#!/usr/bin/env bash
# Rent the wheel gate's GPU pod and write pod_id, ip and port (the pod's
# public SSH endpoint) to $GITHUB_OUTPUT. Run by the validate-cuda job in
# .github/workflows/wheels.yml; python/tests/test_wheel_gate_rent.py runs it
# against a fake curl.
# Usage: RUNPOD_KEY=<key> POD_NAME=<name> IMG=<image:tag> \
#   POD_PUBLIC_KEY_FILE=<path> GITHUB_OUTPUT=<path> bash scripts/wheel_gate_rent.sh
set -euo pipefail

for var in RUNPOD_KEY POD_NAME IMG POD_PUBLIC_KEY_FILE GITHUB_OUTPUT; do
    [ -n "${!var:-}" ] || { echo "error: $var is unset" >&2; exit 2; }
done
PUB="$(cat "$POD_PUBLIC_KEY_FILE")"

create() {
    curl -sf https://rest.runpod.io/v1/pods \
        -H "Authorization: Bearer $RUNPOD_KEY" -H 'content-type: application/json' \
        -d "{\"name\": \"$POD_NAME\", \"imageName\": \"$IMG\",
             \"gpuTypeIds\": [\"NVIDIA L40S\"], \"gpuCount\": 1,
             \"cloudType\": \"SECURE\", \"containerDiskInGb\": 20,
             \"ports\": [\"22/tcp\"], \"env\": {\"PUBLIC_KEY\": \"$PUB\"}}" \
        | jq -r .id
}

# Five attempts, because capacity churns on ~10-minute cycles: two quick tries
# can both land in the same empty window, so failed creates back off 150s to
# spread the attempts across a full cycle. Each successful create polls up to
# 6 minutes on the GraphQL runtime, the only truthful liveness signal.
for attempt in 1 2 3 4 5; do
    POD="$(create || true)"
    if [ -z "${POD}" ] || [ "${POD}" = "null" ]; then
        echo "attempt ${attempt}: pod create call failed"
        sleep 150
        continue
    fi
    echo "attempt ${attempt}: pod ${POD}"
    echo "pod_id=${POD}" >> "$GITHUB_OUTPUT"
    for i in $(seq 1 36); do
        sleep 10
        OUT=$(curl -s https://api.runpod.io/graphql \
            -H "Authorization: Bearer $RUNPOD_KEY" -H 'content-type: application/json' \
            -d "{\"query\":\"query{pod(input:{podId:\\\"$POD\\\"}){runtime{uptimeInSeconds ports{ip isIpPublic privatePort publicPort}}}}\"}")
        IP=$(echo "$OUT" | jq -r '.data.pod.runtime.ports[]? | select(.isIpPublic and .privatePort==22) | .ip' | head -1)
        PORT=$(echo "$OUT" | jq -r '.data.pod.runtime.ports[]? | select(.isIpPublic and .privatePort==22) | .publicPort' | head -1)
        if [ -n "${IP}" ] && [ "${IP}" != "null" ]; then
            echo "ip=${IP}" >> "$GITHUB_OUTPUT"
            echo "port=${PORT}" >> "$GITHUB_OUTPUT"
            exit 0
        fi
    done
    echo "attempt ${attempt}: no public SSH endpoint after 6m; re-rolling"
    curl -s -X DELETE "https://rest.runpod.io/v1/pods/$POD" \
        -H "Authorization: Bearer $RUNPOD_KEY" || true
done
echo "::error::no usable pod after 5 attempts"
exit 1
