//
//  JoltPhysicsPlugin.swift
//  UntoldJoltPhysics
//

import Foundation
import UntoldEngine

/// Stable identifiers owned by the UntoldJoltPhysics package.
public enum JoltPhysicsPluginContract {
    public static let pluginID = "com.untoldengine.joltphysics"
    public static let backendID = "com.untoldengine.joltphysics.backend"
}

/// Jolt Physics packaged as a `PhysicsBackendPlugin`.
public struct JoltPhysicsPlugin: PhysicsBackendPlugin {
    public let manifest = PhysicsBackendPluginManifest(
        id: JoltPhysicsPluginContract.pluginID,
        displayName: "Jolt Physics",
        version: PhysicsBackendVersion(major: 0, minor: 1, patch: 0),
        requiredAPIVersion: .current
    )

    /// The instance handed to the engine, kept so the app can use the
    /// plugin-owned extras (body state, resets).
    public let backend: JoltPhysicsBackend

    public init(backend: JoltPhysicsBackend = JoltPhysicsBackend()) {
        self.backend = backend
    }

    public init(settings: JoltWorldSettings) {
        backend = JoltPhysicsBackend(settings: settings)
    }

    public func makeBackend() -> any PhysicsBackend {
        backend
    }
}

/// Installs the Jolt physics backend. Call once before renderer creation.
/// Returns the live backend on success. When the registry is already locked
/// for runtime (the immersive space was reopened) the installed Jolt backend
/// is reused instead of failing.
@discardableResult
public func registerJoltPhysics(settings: JoltWorldSettings = JoltWorldSettings()) -> JoltPhysicsBackend? {
    let plugin = JoltPhysicsPlugin(settings: settings)
    switch PhysicsBackendRegistry.shared.install(plugin) {
    case .installed, .replaced:
        return plugin.backend
    case let .rejected(failure):
        if failure.registryLocked,
           let existing = PhysicsBackendRegistry.shared.activeBackend() as? JoltPhysicsBackend
        {
            return existing
        }
        print("UntoldJoltPhysics: physics installation rejected:", failure)
        return nil
    }
}
