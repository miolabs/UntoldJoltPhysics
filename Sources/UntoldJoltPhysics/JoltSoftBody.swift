//
//  JoltSoftBody.swift
//  UntoldJoltPhysics
//
//  Soft bodies — particles joined by distance constraints, solved by Jolt's
//  own XPBD, colliding both ways with the rigid bodies — through the
//  plugin's side channel. The engine's physics seam has no soft-body
//  vocabulary, so a game builds one here from its own particle graph and
//  reads the vertex positions back every frame to drive a mesh.
//

import CJoltBridge
import simd
import UntoldEngine

public struct JoltSoftBodyDescriptor: Sendable {
    /// Particle positions, relative to `position`.
    public var vertices: [SIMD3<Float>]
    /// Per particle; 0 pins the particle in place.
    public var inverseMasses: [Float]
    /// Distance constraints between particle indices.
    public var edges: [SIMD2<UInt32>]
    /// Inverse stiffness of the edges, in m/N; 0 is rigid.
    public var compliance: Float = 1e-6
    /// Per-edge compliance overriding `compliance` when set.
    public var edgeCompliances: [Float]? = nil
    /// Optional triangles: with them the body has a surface, so a sphere
    /// cannot slip between particles (and continuous collision applies).
    public var faces: [SIMD3<UInt32>] = []
    public var position: SIMD3<Float> = .zero
    public var layer: UInt32 = 0
    /// Solver iterations per step (Jolt substeps the body by this count).
    public var iterations: UInt32 = 10
    public var linearDamping: Float = 1.0
    /// Collision radius of a particle. Rigid bodies collide with the
    /// particles as spheres, so this radius plus the other body's must
    /// cover half the particle spacing or that body slips through the gaps;
    /// faces alone do not close them.
    public var vertexRadius: Float = 0.004
    public var friction: Float = 0.4
    public var restitution: Float = 0.0
    public var gravityFactor: Float = 1.0

    public init(vertices: [SIMD3<Float>], inverseMasses: [Float], edges: [SIMD2<UInt32>]) {
        self.vertices = vertices
        self.inverseMasses = inverseMasses
        self.edges = edges
    }
}

/// A soft body in the world; remove it with `JoltPhysicsBackend.removeSoftBody`.
public final class JoltSoftBody: @unchecked Sendable {
    let id: ujolt_body_id
    public let vertexCount: Int

    init(id: ujolt_body_id, vertexCount: Int) {
        self.id = id
        self.vertexCount = vertexCount
    }
}

extension JoltPhysicsBackend {
    /// Adds a soft body; nil if the descriptor is malformed (an edge or face
    /// index out of range, a zero-length edge). Frame thread.
    public func addSoftBody(_ descriptor: JoltSoftBodyDescriptor) -> JoltSoftBody? {
        guard !descriptor.vertices.isEmpty, descriptor.inverseMasses.count == descriptor.vertices.count else { return nil }
        var positions: [Float] = []
        positions.reserveCapacity(descriptor.vertices.count * 3)
        for v in descriptor.vertices { positions += [v.x, v.y, v.z] }
        var edges: [UInt32] = []
        edges.reserveCapacity(descriptor.edges.count * 2)
        for e in descriptor.edges { edges += [e.x, e.y] }
        var faces: [UInt32] = []
        faces.reserveCapacity(descriptor.faces.count * 3)
        for f in descriptor.faces { faces += [f.x, f.y, f.z] }
        let compliances = descriptor.edgeCompliances ?? []

        var desc = ujolt_soft_body_desc()
        desc.vertex_count = UInt32(descriptor.vertices.count)
        desc.edge_count = UInt32(descriptor.edges.count)
        desc.face_count = UInt32(descriptor.faces.count)
        desc.compliance = descriptor.compliance
        desc.position = (descriptor.position.x, descriptor.position.y, descriptor.position.z)
        desc.layer = descriptor.layer
        desc.iterations = descriptor.iterations
        desc.linear_damping = descriptor.linearDamping
        desc.vertex_radius = descriptor.vertexRadius
        desc.friction = descriptor.friction
        desc.restitution = descriptor.restitution
        desc.gravity_factor = descriptor.gravityFactor
        // Not an engine entity: its activations look like the environment's.
        desc.user_data = UInt64(Self.environmentEntity)

        let id: ujolt_body_id = positions.withUnsafeBufferPointer { p in
            descriptor.inverseMasses.withUnsafeBufferPointer { m in
                edges.withUnsafeBufferPointer { e in
                    faces.withUnsafeBufferPointer { f in
                        compliances.withUnsafeBufferPointer { c in
                            desc.vertices = p.baseAddress
                            desc.inv_masses = m.baseAddress
                            desc.edges = e.baseAddress
                            desc.faces = f.isEmpty ? nil : f.baseAddress
                            desc.edge_compliances = c.isEmpty ? nil : c.baseAddress
                            return ujolt_world_add_soft_body(worldHandle, &desc)
                        }
                    }
                }
            }
        }
        guard id != UJOLT_INVALID_BODY else { return nil }
        return JoltSoftBody(id: id, vertexCount: Int(ujolt_world_soft_body_vertex_count(worldHandle, id)))
    }

    /// Reads the world-space particle positions into `positions` (resized to
    /// the body's vertex count). Returns the count read. Frame thread,
    /// between steps.
    @discardableResult
    public func readSoftBodyVertices(_ body: JoltSoftBody, into positions: inout [SIMD3<Float>]) -> Int {
        if positions.count != body.vertexCount {
            positions = Array(repeating: .zero, count: body.vertexCount)
        }
        var scratch = [Float](repeating: 0, count: body.vertexCount * 3)
        let read = scratch.withUnsafeMutableBufferPointer { buffer in
            Int(ujolt_world_read_soft_body_vertices(worldHandle, body.id, buffer.baseAddress, UInt32(body.vertexCount)))
        }
        for index in 0 ..< read {
            positions[index] = SIMD3<Float>(scratch[index * 3], scratch[index * 3 + 1], scratch[index * 3 + 2])
        }
        return read
    }

    public func removeSoftBody(_ body: JoltSoftBody) {
        ujolt_world_remove_body(worldHandle, body.id)
    }
}
