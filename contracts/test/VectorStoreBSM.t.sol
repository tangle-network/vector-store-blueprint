// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.26;

import "forge-std/Test.sol";
import { ERC1967Proxy } from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import { VectorStoreBSM } from "../src/VectorStoreBSM.sol";
import { BlueprintServiceManagerBase } from "tnt-core/BlueprintServiceManagerBase.sol";

contract VectorStoreBSMTest is Test {
    VectorStoreBSM public bsm;
    VectorStoreBSM public impl;

    address public owner = makeAddr("owner");
    address public tangleCore = makeAddr("tangleCore");
    address public paymentToken = makeAddr("paymentToken");
    address public operator1 = makeAddr("operator1");
    address public operator2 = makeAddr("operator2");
    address public nobody = makeAddr("nobody");

    function setUp() public {
        impl = new VectorStoreBSM();
        ERC1967Proxy proxy =
            new ERC1967Proxy(address(impl), abi.encodeCall(VectorStoreBSM.initialize, (paymentToken)));
        bsm = VectorStoreBSM(payable(address(proxy)));

        // Wire up BSM base: sets blueprintOwner and tangleCore
        bsm.onBlueprintCreated(1, owner, tangleCore);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // INITIALIZATION
    // ═══════════════════════════════════════════════════════════════════════════

    function test_initialize_setsPaymentToken() public view {
        assertEq(bsm.paymentToken(), paymentToken);
    }

    function test_initialize_setsBlueprintOwner() public view {
        assertEq(bsm.blueprintOwner(), owner);
    }

    function test_initialize_setsDefaultPricing() public view {
        VectorStoreBSM.VectorStorePricing memory p = bsm.getPricing();
        assertEq(p.pricePerGbMonth, 100_000);
        assertEq(p.pricePerKUpserts, 10_000);
        assertEq(p.pricePerKQueries, 5_000);
        assertEq(p.maxCollections, 100);
        assertEq(p.maxVectorsPerCollection, 10_000_000);
    }

    function test_initialize_cannotReinitialize() public {
        vm.expectRevert();
        bsm.initialize(address(0));
    }

    function test_onBlueprintCreated_cannotBeCalledTwice() public {
        vm.expectRevert(BlueprintServiceManagerBase.AlreadyInitialized.selector);
        bsm.onBlueprintCreated(2, makeAddr("other"), makeAddr("otherTangle"));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PRICING CONFIGURATION
    // ═══════════════════════════════════════════════════════════════════════════

    function test_configurePricing_setsValues() public {
        vm.prank(owner);
        bsm.configurePricing(200_000, 20_000, 10_000, 50, 5_000_000);

        VectorStoreBSM.VectorStorePricing memory p = bsm.getPricing();
        assertEq(p.pricePerGbMonth, 200_000);
        assertEq(p.pricePerKUpserts, 20_000);
        assertEq(p.pricePerKQueries, 10_000);
        assertEq(p.maxCollections, 50);
        assertEq(p.maxVectorsPerCollection, 5_000_000);
    }

    function test_configurePricing_emitsEvent() public {
        vm.expectEmit(true, true, true, true);
        emit VectorStoreBSM.PricingConfigured(200_000, 20_000, 10_000, 50, 5_000_000);

        vm.prank(owner);
        bsm.configurePricing(200_000, 20_000, 10_000, 50, 5_000_000);
    }

    function test_configurePricing_revertsForNonOwner() public {
        vm.prank(nobody);
        vm.expectRevert(
            abi.encodeWithSelector(BlueprintServiceManagerBase.OnlyBlueprintOwnerAllowed.selector, nobody, owner)
        );
        bsm.configurePricing(200_000, 20_000, 10_000, 50, 5_000_000);
    }

    function test_configurePricing_canSetToZero() public {
        vm.prank(owner);
        bsm.configurePricing(0, 0, 0, 0, 0);

        VectorStoreBSM.VectorStorePricing memory p = bsm.getPricing();
        assertEq(p.pricePerGbMonth, 0);
        assertEq(p.pricePerKUpserts, 0);
        assertEq(p.pricePerKQueries, 0);
        assertEq(p.maxCollections, 0);
        assertEq(p.maxVectorsPerCollection, 0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // OPERATOR REGISTRATION
    // ═══════════════════════════════════════════════════════════════════════════

    function test_onRegister_recordsOperator() public {
        bytes memory inputs = abi.encode("qdrant", "https://qdrant.example.com:6334");

        vm.prank(tangleCore);
        bsm.onRegister(operator1, inputs);

        VectorStoreBSM.OperatorInfo memory info = bsm.getOperator(operator1);
        assertEq(info.backend, "qdrant");
        assertEq(info.endpoint, "https://qdrant.example.com:6334");
        assertEq(info.collectionCount, 0);
        assertTrue(info.active);
    }

    function test_onRegister_emitsEvent() public {
        bytes memory inputs = abi.encode("qdrant", "https://qdrant.example.com:6334");

        vm.expectEmit(true, true, true, true);
        emit VectorStoreBSM.OperatorRegistered(operator1, "qdrant", "https://qdrant.example.com:6334");

        vm.prank(tangleCore);
        bsm.onRegister(operator1, inputs);
    }

    function test_onRegister_addsToOperatorSet() public {
        bytes memory inputs = abi.encode("chromadb", "http://chroma:8000");

        vm.prank(tangleCore);
        bsm.onRegister(operator1, inputs);

        assertEq(bsm.getOperatorCount(), 1);
        assertEq(bsm.getOperatorAt(0), operator1);
    }

    function test_onRegister_multipleOperators() public {
        vm.prank(tangleCore);
        bsm.onRegister(operator1, abi.encode("qdrant", "https://op1:6334"));

        vm.prank(tangleCore);
        bsm.onRegister(operator2, abi.encode("inmemory", "http://op2:8080"));

        assertEq(bsm.getOperatorCount(), 2);
        assertTrue(bsm.isOperatorActive(operator1));
        assertTrue(bsm.isOperatorActive(operator2));
    }

    function test_onRegister_revertsForNonTangle() public {
        bytes memory inputs = abi.encode("qdrant", "https://qdrant:6334");

        vm.prank(nobody);
        vm.expectRevert(
            abi.encodeWithSelector(BlueprintServiceManagerBase.OnlyTangleAllowed.selector, nobody, tangleCore)
        );
        bsm.onRegister(operator1, inputs);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // OPERATOR UNREGISTRATION
    // ═══════════════════════════════════════════════════════════════════════════

    function test_onUnregister_marksInactive() public {
        vm.prank(tangleCore);
        bsm.onRegister(operator1, abi.encode("qdrant", "https://qdrant:6334"));

        vm.prank(tangleCore);
        bsm.onUnregister(operator1);

        assertFalse(bsm.isOperatorActive(operator1));
    }

    function test_onUnregister_removesFromSet() public {
        vm.prank(tangleCore);
        bsm.onRegister(operator1, abi.encode("qdrant", "https://qdrant:6334"));

        vm.prank(tangleCore);
        bsm.onUnregister(operator1);

        assertEq(bsm.getOperatorCount(), 0);
    }

    function test_onUnregister_preservesOtherOperators() public {
        vm.prank(tangleCore);
        bsm.onRegister(operator1, abi.encode("qdrant", "https://op1:6334"));
        vm.prank(tangleCore);
        bsm.onRegister(operator2, abi.encode("inmemory", "http://op2:8080"));

        vm.prank(tangleCore);
        bsm.onUnregister(operator1);

        assertEq(bsm.getOperatorCount(), 1);
        assertTrue(bsm.isOperatorActive(operator2));
        assertFalse(bsm.isOperatorActive(operator1));
    }

    function test_onUnregister_revertsForNonTangle() public {
        vm.prank(tangleCore);
        bsm.onRegister(operator1, abi.encode("qdrant", "https://qdrant:6334"));

        vm.prank(nobody);
        vm.expectRevert(
            abi.encodeWithSelector(BlueprintServiceManagerBase.OnlyTangleAllowed.selector, nobody, tangleCore)
        );
        bsm.onUnregister(operator1);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ACCESS CONTROL
    // ═══════════════════════════════════════════════════════════════════════════

    function test_onRequest_revertsForNonTangle() public {
        address[] memory operators = new address[](0);
        vm.prank(nobody);
        vm.expectRevert(
            abi.encodeWithSelector(BlueprintServiceManagerBase.OnlyTangleAllowed.selector, nobody, tangleCore)
        );
        bsm.onRequest(0, address(0), operators, "", 0, address(0), 0);
    }

    function test_onRequest_succeedsFromTangle() public {
        address[] memory operators = new address[](0);
        vm.prank(tangleCore);
        bsm.onRequest(0, address(0), operators, "", 0, address(0), 0);
    }

    function test_authorizeUpgrade_revertsForNonOwner() public {
        vm.prank(nobody);
        vm.expectRevert(
            abi.encodeWithSelector(BlueprintServiceManagerBase.OnlyBlueprintOwnerAllowed.selector, nobody, owner)
        );
        bsm.upgradeToAndCall(address(impl), "");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    function test_getOperator_returnsEmptyForUnregistered() public view {
        VectorStoreBSM.OperatorInfo memory info = bsm.getOperator(nobody);
        assertEq(info.backend, "");
        assertEq(info.endpoint, "");
        assertEq(info.collectionCount, 0);
        assertFalse(info.active);
    }

    function test_isOperatorActive_falseForUnregistered() public view {
        assertFalse(bsm.isOperatorActive(nobody));
    }

    function test_getOperatorCount_zeroInitially() public view {
        assertEq(bsm.getOperatorCount(), 0);
    }

    function test_getPricing_returnsCurrentValues() public {
        vm.prank(owner);
        bsm.configurePricing(1, 2, 3, 4, 5);

        VectorStoreBSM.VectorStorePricing memory p = bsm.getPricing();
        assertEq(p.pricePerGbMonth, 1);
        assertEq(p.pricePerKUpserts, 2);
        assertEq(p.pricePerKQueries, 3);
        assertEq(p.maxCollections, 4);
        assertEq(p.maxVectorsPerCollection, 5);
    }

    function test_publicPricingGetter_matchesGetPricing() public view {
        (uint64 gbMonth, uint64 kUpserts, uint64 kQueries, uint32 maxColl, uint64 maxVec) = bsm.pricing();
        VectorStoreBSM.VectorStorePricing memory p = bsm.getPricing();
        assertEq(gbMonth, p.pricePerGbMonth);
        assertEq(kUpserts, p.pricePerKUpserts);
        assertEq(kQueries, p.pricePerKQueries);
        assertEq(maxColl, p.maxCollections);
        assertEq(maxVec, p.maxVectorsPerCollection);
    }

    function test_getOperatorAt_revertsOutOfBounds() public {
        vm.expectRevert();
        bsm.getOperatorAt(0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // FUZZ TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    function testFuzz_configurePricing(
        uint64 gbMonth,
        uint64 kUpserts,
        uint64 kQueries,
        uint32 maxColl,
        uint64 maxVec
    ) public {
        vm.prank(owner);
        bsm.configurePricing(gbMonth, kUpserts, kQueries, maxColl, maxVec);

        VectorStoreBSM.VectorStorePricing memory p = bsm.getPricing();
        assertEq(p.pricePerGbMonth, gbMonth);
        assertEq(p.pricePerKUpserts, kUpserts);
        assertEq(p.pricePerKQueries, kQueries);
        assertEq(p.maxCollections, maxColl);
        assertEq(p.maxVectorsPerCollection, maxVec);
    }

    function testFuzz_onRegister_arbitraryBackend(string calldata backend, string calldata endpoint) public {
        bytes memory inputs = abi.encode(backend, endpoint);

        vm.prank(tangleCore);
        bsm.onRegister(operator1, inputs);

        VectorStoreBSM.OperatorInfo memory info = bsm.getOperator(operator1);
        assertEq(info.backend, backend);
        assertEq(info.endpoint, endpoint);
        assertTrue(info.active);
    }
}
