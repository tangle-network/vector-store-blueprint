// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { EnumerableSet } from "@openzeppelin/contracts/utils/structs/EnumerableSet.sol";
import { Initializable } from "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import { UUPSUpgradeable } from "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import { BlueprintServiceManagerBase } from "tnt-core/BlueprintServiceManagerBase.sol";

/// @title VectorStoreBSM — Blueprint Service Manager for hosted vector storage + similarity search.
/// @notice Manages operator registration with pricing across three dimensions:
///         storage (per-GB-month), writes (per-1K upserts), and queries (per-1K searches).
///         Operators run Qdrant, ChromaDB, or in-memory backends.
contract VectorStoreBSM is Initializable, UUPSUpgradeable, BlueprintServiceManagerBase {
    using EnumerableSet for EnumerableSet.AddressSet;

    // ── Errors ──────────────────────────────────────────────────────────

    error MaxCollectionsExceeded(uint32 requested, uint32 maxAllowed);
    error MaxVectorsExceeded(uint64 requested, uint64 maxAllowed);

    // ── Events ──────────────────────────────────────────────────────────

    event OperatorRegistered(address indexed operator, string backend, string endpoint);
    event PricingConfigured(
        uint64 pricePerGbMonth,
        uint64 pricePerKUpserts,
        uint64 pricePerKQueries,
        uint32 maxCollections,
        uint64 maxVectorsPerCollection
    );

    // ── Storage ─────────────────────────────────────────────────────────

    struct VectorStorePricing {
        uint64 pricePerGbMonth;
        uint64 pricePerKUpserts;
        uint64 pricePerKQueries;
        uint32 maxCollections;
        uint64 maxVectorsPerCollection;
    }

    struct OperatorInfo {
        string backend;
        string endpoint;
        uint32 collectionCount;
        bool active;
    }

    VectorStorePricing public pricing;
    mapping(address => OperatorInfo) public operatorInfo;
    EnumerableSet.AddressSet private _operators;
    /// The accepted payment token (e.g. USDC wrapped via VAnchor).
    address public paymentToken;

    // ── Initialization ──────────────────────────────────────────────────

    function initialize(address _paymentToken) external initializer {
        __UUPSUpgradeable_init();
        paymentToken = _paymentToken;
        pricing = VectorStorePricing({
            pricePerGbMonth: 100_000,
            pricePerKUpserts: 10_000,
            pricePerKQueries: 5_000,
            maxCollections: 100,
            maxVectorsPerCollection: 10_000_000
        });
    }

    function _authorizeUpgrade(address) internal override onlyBlueprintOwner {}

    // ── Admin ───────────────────────────────────────────────────────────

    function configurePricing(
        uint64 pricePerGbMonth,
        uint64 pricePerKUpserts,
        uint64 pricePerKQueries,
        uint32 maxCollections,
        uint64 maxVectorsPerCollection
    ) external onlyBlueprintOwner {
        pricing = VectorStorePricing({
            pricePerGbMonth: pricePerGbMonth,
            pricePerKUpserts: pricePerKUpserts,
            pricePerKQueries: pricePerKQueries,
            maxCollections: maxCollections,
            maxVectorsPerCollection: maxVectorsPerCollection
        });
        emit PricingConfigured(
            pricePerGbMonth, pricePerKUpserts, pricePerKQueries, maxCollections, maxVectorsPerCollection
        );
    }

    // ── Lifecycle Hooks ─────────────────────────────────────────────────

    /// @param registrationInputs abi.encode(string backend, string endpoint)
    function onRegister(
        address operator,
        bytes calldata registrationInputs
    ) external payable override onlyFromTangle {
        (string memory backend, string memory endpoint) =
            abi.decode(registrationInputs, (string, string));

        operatorInfo[operator] = OperatorInfo({
            backend: backend,
            endpoint: endpoint,
            collectionCount: 0,
            active: true
        });
        _operators.add(operator);

        emit OperatorRegistered(operator, backend, endpoint);
    }

    function onUnregister(
        address operator
    ) external override onlyFromTangle {
        operatorInfo[operator].active = false;
        _operators.remove(operator);
    }

    function onRequest(
        uint64,
        address,
        address[] calldata,
        bytes calldata,
        uint64,
        address,
        uint256
    ) external payable override onlyFromTangle {
        // Service requests are handled via HTTP (x402), not on-chain jobs.
    }

    // ── Views ───────────────────────────────────────────────────────────

    function getOperator(address operator) external view returns (OperatorInfo memory) {
        return operatorInfo[operator];
    }

    function getOperatorCount() external view returns (uint256) {
        return _operators.length();
    }

    function getOperatorAt(uint256 index) external view returns (address) {
        return _operators.at(index);
    }

    function isOperatorActive(address operator) external view returns (bool) {
        return operatorInfo[operator].active;
    }

    function getPricing() external view returns (VectorStorePricing memory) {
        return pricing;
    }
}
