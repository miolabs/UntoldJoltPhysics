//
//  JoltPhysicsPluginTests.swift
//  UntoldJoltPhysicsTests
//
//  Manifest and registration behavior against the engine's registry. The
//  registry is process-global and locks for good on the first simulated
//  substep, so these tests never step through the coordinator; they install
//  and uninstall explicitly and leave the registry empty behind them.
//

import simd
@testable import UntoldJoltPhysics
import UntoldEngine
import XCTest

final class JoltPhysicsPluginTests: XCTestCase {
    override func tearDown() {
        PhysicsBackendRegistry.shared.uninstall(id: JoltPhysicsPluginContract.pluginID)
        super.tearDown()
    }

    func testManifestDeclaresTheCurrentAPIVersion() {
        let plugin = JoltPhysicsPlugin()
        XCTAssertEqual(plugin.manifest.id, JoltPhysicsPluginContract.pluginID)
        XCTAssertEqual(plugin.manifest.requiredAPIVersion, PhysicsBackendAPIVersion.current)
        XCTAssertEqual(plugin.makeBackend().id, JoltPhysicsPluginContract.backendID)
        XCTAssertTrue(plugin.makeBackend() === plugin.backend, "The engine gets the instance the app keeps")
    }

    func testInstallUninstallRoundTrip() {
        let plugin = JoltPhysicsPlugin()
        guard case .installed = PhysicsBackendRegistry.shared.install(plugin) else {
            return XCTFail("First install must succeed")
        }
        XCTAssertEqual(PhysicsBackendRegistry.shared.activeManifest()?.id, JoltPhysicsPluginContract.pluginID)
        XCTAssertTrue(PhysicsBackendRegistry.shared.activeBackend() === plugin.backend)

        XCTAssertTrue(PhysicsBackendRegistry.shared.uninstall(id: JoltPhysicsPluginContract.pluginID))
        XCTAssertNil(PhysicsBackendRegistry.shared.activeBackend())
        XCTAssertFalse(PhysicsBackendRegistry.shared.uninstall(id: JoltPhysicsPluginContract.pluginID), "Second uninstall is a no-op")
    }

    func testReinstallingTheSamePluginReplaces() {
        let first = JoltPhysicsPlugin()
        let second = JoltPhysicsPlugin()
        guard case .installed = PhysicsBackendRegistry.shared.install(first) else {
            return XCTFail("First install must succeed")
        }
        guard case .replaced = PhysicsBackendRegistry.shared.install(second) else {
            return XCTFail("Same plugin ID must replace")
        }
        XCTAssertTrue(PhysicsBackendRegistry.shared.activeBackend() === second.backend)
    }

    func testRegistrationHelperReturnsTheLiveBackend() {
        let backend = registerJoltPhysics()
        XCTAssertNotNil(backend)
        XCTAssertTrue(PhysicsBackendRegistry.shared.activeBackend() === backend)
        XCTAssertEqual(backend?.bodyCount, 0)
    }

    func testCustomSettingsReachTheWorld() {
        var settings = JoltWorldSettings()
        settings.workerThreads = 0
        settings.maxBodies = 8
        let backend = JoltPhysicsBackend(settings: settings)
        backend.configure(PhysicsWorldConfiguration())
        for index in 0 ..< 10 {
            backend.didAddBody(entity: EntityID(index + 1), descriptor: PhysicsBodyDescriptor(
                motionType: .dynamic,
                collider: PhysicsColliderDescriptor(shape: .sphere(radius: 0.1)),
                position: simd_float3(Float(index), 1, 0)
            ))
        }
        XCTAssertEqual(backend.bodyCount, 8, "maxBodies reached the world: Jolt refuses the ninth body")
        backend.step(deltaTime: 1.0 / 60.0)
        XCTAssertEqual(backend.bodyCount, 8)
    }
}
