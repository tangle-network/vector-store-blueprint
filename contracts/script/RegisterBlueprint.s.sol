// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { Script, console2 } from "forge-std/Script.sol";
import { ERC1967Proxy } from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import { Types } from "tnt-core/libraries/Types.sol";
import { VectorStoreBSM } from "../src/VectorStoreBSM.sol";

/// @notice Minimal interface for Tangle blueprint registration.
interface ITangle {
    function createBlueprint(Types.BlueprintDefinition calldata def) external returns (uint64);
}

/// @title RegisterBlueprint
/// @notice Deploys VectorStoreBSM (impl + UUPS proxy + initialize) and registers
///         the vector-store blueprint on Tangle in a single broadcast.
/// @dev    Run via `deploy/register-blueprint.sh`, which wraps:
///           forge script contracts/script/RegisterBlueprint.s.sol
///             --rpc-url $RPC_URL --broadcast --slow
///
///         The BSM is upgradeable (UUPS) — `initialize(paymentToken)` is invoked
///         atomically inside the ERC1967Proxy constructor so the freshly minted
///         proxy is fully wired before Tangle.createBlueprint records it as the
///         blueprint manager.
contract RegisterBlueprint is Script {
    // ─────────────────────────────────────────────────────────────────────────
    // Defaults — overridable via env vars for non-anvil chains.
    // ─────────────────────────────────────────────────────────────────────────

    // Anvil well-known deployer key (default when no PRIVATE_KEY env is set).
    uint256 constant DEFAULT_DEPLOYER_KEY =
        0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80;

    // Tangle protocol address on a LocalTestnet anvil snapshot. For real chains
    // (Base Sepolia, mainnet) pass TANGLE_CORE via env.
    address constant DEFAULT_TANGLE = 0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9;

    // USDC on Base Sepolia. Vector-store operators settle storage / write /
    // query fees in this token. For other networks pass PAYMENT_TOKEN via env.
    address constant DEFAULT_PAYMENT_TOKEN = 0x036CbD53842c5426634e7929541eC2318f3dCF7e;

    function run() external {
        uint256 deployerKey = vm.envOr("PRIVATE_KEY", DEFAULT_DEPLOYER_KEY);
        address tangleAddr = vm.envOr("TANGLE_CORE", DEFAULT_TANGLE);
        address paymentToken = vm.envOr("PAYMENT_TOKEN", DEFAULT_PAYMENT_TOKEN);

        ITangle tangle = ITangle(tangleAddr);

        vm.startBroadcast(deployerKey);

        // ── Deploy VectorStoreBSM (UUPS impl + proxy + initialize) ──────────
        VectorStoreBSM impl = new VectorStoreBSM();
        ERC1967Proxy proxy = new ERC1967Proxy(
            address(impl),
            abi.encodeCall(VectorStoreBSM.initialize, (paymentToken))
        );
        VectorStoreBSM bsm = VectorStoreBSM(payable(address(proxy)));

        // ── Register on Tangle ──────────────────────────────────────────────
        uint64 blueprintId = tangle.createBlueprint(_buildDefinition(address(bsm)));

        vm.stopBroadcast();

        // ── Output for bash wrapper parsing ─────────────────────────────────
        console2.log("DEPLOY_VECTOR_STORE_BSM_IMPL=%s", vm.toString(address(impl)));
        console2.log("DEPLOY_VECTOR_STORE_BSM_PROXY=%s", vm.toString(address(bsm)));
        console2.log("DEPLOY_VECTOR_STORE_BLUEPRINT_ID=%s", vm.toString(blueprintId));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Blueprint Definition builder
    // ═════════════════════════════════════════════════════════════════════════

    function _buildDefinition(address manager) internal pure returns (Types.BlueprintDefinition memory def) {
        def.metadataUri = "https://github.com/tangle-network/vector-store-blueprint";
        // metadataHash is a digest of the canonical metadata JSON. Until that
        // payload is pinned via IPFS, derive it from the metadataUri so the
        // value is deterministic + traceable.
        def.metadataHash = keccak256(bytes(def.metadataUri));
        def.manager = manager;
        def.masterManagerRevision = 0;
        def.hasConfig = true;

        // Vector-store has no on-chain jobs — collection CRUD, upserts, and
        // similarity queries are served over HTTP (x402) and metered off-chain.
        // The blueprint is registered for operator/service membership and for
        // the metered subscription rails, not for `submitJob`-style RPC.
        // EventDriven keeps the door open for future per-call settlement.
        def.config = Types.BlueprintConfig({
            membership: Types.MembershipModel.Dynamic,
            pricing: Types.PricingModel.EventDriven,
            minOperators: 1,
            maxOperators: 0, // unbounded
            subscriptionRate: 0,
            subscriptionInterval: 0,
            eventRate: 0 // metering happens off-chain in tsUSD-denominated units
        });

        def.metadata = Types.BlueprintMetadata({
            name: "Vector Store Blueprint",
            description: "Hosted vector storage and similarity search for RAG workloads on Tangle",
            author: "Tangle Network",
            category: "AI Infrastructure",
            codeRepository: "https://github.com/tangle-network/vector-store-blueprint",
            logo: "",
            website: "https://tangle.tools",
            license: "MIT OR Apache-2.0",
            profilingData: ""
        });

        // Tangle's `BlueprintsCreate._validateBlueprintDefinition` rejects any
        // blueprint with an empty `jobs` array (reverts `InvalidState()`), so
        // declare a single sentinel job entry to satisfy the precondition.
        //
        // Vector-store has no real on-chain jobs — collection CRUD, upserts,
        // and similarity queries are served over HTTP (x402) and metered
        // off-chain. The full HTTP surface (`/v1/collections`,
        // `/v1/collections/:name/upsert`, …) is documented in the manifest
        // blueprint-metadata.json. This sentinel exists purely to clear the
        // on-chain validator; nothing dispatches against it.
        def.jobs = new Types.JobDefinition[](1);
        def.jobs[0] = Types.JobDefinition({
            name: "noop",
            description: "Sentinel job; vector-store dispatches over HTTP, not on-chain.",
            metadataUri: "https://github.com/tangle-network/vector-store-blueprint",
            paramsSchema: "",
            resultSchema: ""
        });

        def.registrationSchema = "";
        def.requestSchema = "";

        def.sources = new Types.BlueprintSource[](1);
        Types.BlueprintBinary[] memory bins = new Types.BlueprintBinary[](1);
        bins[0] = Types.BlueprintBinary({
            arch: Types.BlueprintArchitecture.Amd64,
            os: Types.BlueprintOperatingSystem.Linux,
            name: "vector-store",
            sha256: bytes32(uint256(0xdeadbeef))
        });
        def.sources[0] = Types.BlueprintSource({
            kind: Types.BlueprintSourceKind.Native,
            container: Types.ImageRegistrySource("", "", ""),
            wasm: Types.WasmSource(Types.WasmRuntime.Unknown, Types.BlueprintFetcherKind.None, "", ""),
            native: Types.NativeSource(
                Types.BlueprintFetcherKind.None,
                "file:///target/release/vector-store",
                "./target/release/vector-store"
            ),
            testing: Types.TestingSource("vector-store", "vector-store", "."),
            binaries: bins
        });

        def.supportedMemberships = new Types.MembershipModel[](2);
        def.supportedMemberships[0] = Types.MembershipModel.Dynamic;
        def.supportedMemberships[1] = Types.MembershipModel.Fixed;
    }
}
