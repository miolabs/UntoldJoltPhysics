# UntoldJoltPhysics

[Jolt Physics](https://github.com/jrouwe/JoltPhysics) as a physics backend plugin for [Untold Engine](https://github.com/untoldengine/UntoldEngine), behind the engine's `PhysicsBackend` plugin seam.

Jolt is not vendored here: it is pulled as its own Swift package, source only, no binaries. This repository is the C ABI bridge and Swift adapter that let the engine drive Jolt through `PhysicsBackend`, `PhysicsBackendPlugin`, and friends.

Status: repository just created; code is migrating over from [untoldengine/UntoldArcade#27](https://github.com/untoldengine/UntoldArcade/pull/27), where it currently lives under `Plugins/UntoldJoltPhysics/`.

## License

Licensed under the [Mozilla Public License 2.0](LICENSE), matching Untold Engine.

Jolt Physics itself is MIT-licensed; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) once the code lands.
