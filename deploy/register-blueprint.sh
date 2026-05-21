#!/usr/bin/env bash
# Register the vector-store blueprint on Tangle.
#
# Single-shot flow: deploys VectorStoreBSM (UUPS impl + ERC1967 proxy +
# initialize) AND calls Tangle.createBlueprint in the same broadcast via
# `contracts/script/RegisterBlueprint.s.sol`.
#
# Pattern matches llm-inference-blueprint's register-blueprint.sh (PR #7).
# Vector-store has no `deploy/definition.json` and the BSM is upgradeable, so
# we cannot use the cargo-tangle `--definition` path that voice-inference uses
# (that flow assumes a plain `forge create` works against the BSM constructor).
#
# Prerequisites:
#   - forge installed
#   - Deployer wallet funded on the target network
#
# Usage (Base Sepolia, against the already-deployed Tangle protocol):
#
#   export PRIVATE_KEY=0x...
#   export RPC_URL=https://sepolia.base.org
#   export TANGLE_CORE=0xC9b0716a187072be0f38A5D972392C6479b9Cfe3
#   export PAYMENT_TOKEN=0x036CbD53842c5426634e7929541eC2318f3dCF7e  # USDC sepolia
#   ./deploy/register-blueprint.sh
#
# Local anvil (LocalTestnet snapshot):
#
#   export RPC_URL=http://127.0.0.1:8545
#   ./deploy/register-blueprint.sh   # uses anvil deployer key + Tangle/USDC defaults
#
# Optional overrides:
#   BACKEND     Vector backend an operator will advertise (default: qdrant)
#   ENDPOINT    Operator HTTP endpoint (default: https://your-operator.example.com)
#
# Outputs (parsed by deployment scripts, do not change without coordinating):
#   DEPLOY_VECTOR_STORE_BSM_IMPL=<address>
#   DEPLOY_VECTOR_STORE_BSM_PROXY=<address>
#   DEPLOY_VECTOR_STORE_BLUEPRINT_ID=<u64>

set -euo pipefail

: "${RPC_URL:?Set RPC_URL}"
: "${PRIVATE_KEY:?Set PRIVATE_KEY}"

BACKEND="${BACKEND:-qdrant}"
ENDPOINT="${ENDPOINT:-https://your-operator.example.com}"

echo "=== Vector-Store Blueprint Registration ==="
echo "Network:     $(cast chain-id --rpc-url "$RPC_URL")"
echo "Deployer:    $(cast wallet address --private-key "$PRIVATE_KEY")"
echo "Tangle Core: ${TANGLE_CORE:-<default from RegisterBlueprint.s.sol>}"
echo "Payment:     ${PAYMENT_TOKEN:-<default USDC sepolia>}"
echo "Backend:     $BACKEND"
echo "Endpoint:    $ENDPOINT"
echo ""

cd "$(dirname "$0")/../contracts"

# Deploy BSM (impl + proxy + initialize) AND register the blueprint in one
# forge-script broadcast.
DEPLOY_OUTPUT=$(PRIVATE_KEY="$PRIVATE_KEY" \
    TANGLE_CORE="${TANGLE_CORE:-}" \
    PAYMENT_TOKEN="${PAYMENT_TOKEN:-}" \
    forge script script/RegisterBlueprint.s.sol \
        --rpc-url "$RPC_URL" \
        --broadcast --slow)

echo "$DEPLOY_OUTPUT"

# Extract the BSM proxy address + blueprint ID for downstream scripts.
BSM_ADDRESS=$(echo "$DEPLOY_OUTPUT" | grep -oE 'DEPLOY_VECTOR_STORE_BSM_PROXY=0x[0-9a-fA-F]+' | tail -1 | cut -d= -f2)
BLUEPRINT_ID=$(echo "$DEPLOY_OUTPUT" | grep -oE 'DEPLOY_VECTOR_STORE_BLUEPRINT_ID=[0-9]+' | tail -1 | cut -d= -f2)

if [ -z "$BSM_ADDRESS" ] || [ -z "$BLUEPRINT_ID" ]; then
    echo "ERROR: failed to extract addresses from forge output"
    exit 1
fi

echo ""
echo "=== Blueprint registered ==="
echo "Blueprint ID:           $BLUEPRINT_ID"
echo "VectorStoreBSM proxy:   $BSM_ADDRESS"
echo ""

# Operator registration is a separate step (per-operator). Encode the
# registration inputs so the operator can call Tangle.registerOperator with
# the right calldata.
#
# VectorStoreBSM.onRegister expects abi.encode(string backend, string endpoint)
# — the BSM stores those alongside the operator's `active` flag.
REG_INPUTS=$(cast abi-encode "f(string,string)" "$BACKEND" "$ENDPOINT")

echo "Operator registration inputs (use these to register an operator):"
echo "  $REG_INPUTS"
echo ""
echo "To register an operator now:"
echo "  cast send ${TANGLE_CORE:-<TANGLE_CORE>} \\"
echo "    'registerOperator(uint64,bytes)' $BLUEPRINT_ID $REG_INPUTS \\"
echo "    --rpc-url $RPC_URL --private-key \$OPERATOR_KEY"
