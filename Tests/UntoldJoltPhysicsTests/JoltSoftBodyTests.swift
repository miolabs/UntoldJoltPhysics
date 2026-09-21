//
//  JoltSoftBodyTests.swift
//  UntoldJoltPhysics
//

import simd
import UntoldEngine
@testable import UntoldJoltPhysics
import XCTest

final class JoltSoftBodyTests: XCTestCase {
    private let step: Float = 1.0 / 60.0

    private func makeBackend() -> JoltPhysicsBackend {
        var settings = JoltWorldSettings()
        settings.workerThreads = 0
        let backend = JoltPhysicsBackend(settings: settings)
        backend.configure(PhysicsWorldConfiguration())
        backend.didAddBody(entity: 1000, descriptor: PhysicsBodyDescriptor(
            motionType: .static,
            collider: PhysicsColliderDescriptor(shape: .box(halfExtents: simd_float3(50, 0.5, 50)), friction: 0.5, restitution: 0.0),
            position: simd_float3(0, -0.5, 0)
        ))
        return backend
    }

    private func advance(_ backend: JoltPhysicsBackend, seconds: Float) {
        var elapsed: Float = 0
        while elapsed < seconds {
            backend.step(deltaTime: step)
            elapsed += step
        }
    }

    /// A horizontal square sheet of `n`×`n` particles, `spacing` apart, at
    /// height `y`, its four corners pinned; triangulated so it has a surface.
    private func sheet(n: Int, spacing: Float, y: Float) -> JoltSoftBodyDescriptor {
        var vertices: [SIMD3<Float>] = []
        var inverseMasses: [Float] = []
        let half = Float(n - 1) * spacing * 0.5
        for row in 0 ..< n {
            for column in 0 ..< n {
                vertices.append(SIMD3<Float>(Float(column) * spacing - half, y, Float(row) * spacing - half))
                let corner = (row == 0 || row == n - 1) && (column == 0 || column == n - 1)
                inverseMasses.append(corner ? 0 : 1.0 / 0.05)
            }
        }
        var edges: [SIMD2<UInt32>] = []
        var faces: [SIMD3<UInt32>] = []
        func index(_ row: Int, _ column: Int) -> UInt32 { UInt32(row * n + column) }
        for row in 0 ..< n {
            for column in 0 ..< n {
                if column + 1 < n { edges.append(SIMD2(index(row, column), index(row, column + 1))) }
                if row + 1 < n { edges.append(SIMD2(index(row, column), index(row + 1, column))) }
                if row + 1 < n, column + 1 < n {
                    edges.append(SIMD2(index(row, column), index(row + 1, column + 1)))
                    faces.append(SIMD3(index(row, column), index(row + 1, column), index(row, column + 1)))
                    faces.append(SIMD3(index(row + 1, column), index(row + 1, column + 1), index(row, column + 1)))
                }
            }
        }
        var descriptor = JoltSoftBodyDescriptor(vertices: vertices, inverseMasses: inverseMasses, edges: edges)
        descriptor.faces = faces
        descriptor.compliance = 1e-6
        descriptor.iterations = 10
        // Particles collide as spheres: their radius plus the ball's must
        // cover half the spacing, or the ball slips between them once the
        // sheet starts to oscillate.
        descriptor.vertexRadius = 0.03
        return descriptor
    }

    func testRopeHangsStraightDownFromItsPin() {
        let backend = makeBackend()
        // Two particles: the first pinned, the second held out sideways.
        var descriptor = JoltSoftBodyDescriptor(
            vertices: [SIMD3<Float>(0, 1.5, 0), SIMD3<Float>(0.3, 1.5, 0)],
            inverseMasses: [0, 1.0 / 0.01],
            edges: [SIMD2(0, 1)]
        )
        descriptor.compliance = 0
        descriptor.linearDamping = 2.0
        let rope = try! XCTUnwrap(backend.addSoftBody(descriptor))
        XCTAssertEqual(rope.vertexCount, 2)
        advance(backend, seconds: 3.0)
        var positions: [SIMD3<Float>] = []
        XCTAssertEqual(backend.readSoftBodyVertices(rope, into: &positions), 2)
        XCTAssertEqual(simd_length(positions[0] - SIMD3<Float>(0, 1.5, 0)), 0, accuracy: 1e-4, "The pin does not move")
        XCTAssertEqual(simd_length(positions[1] - positions[0]), 0.3, accuracy: 0.02, "The rope keeps its length")
        XCTAssertLessThan(positions[1].y, 1.5 - 0.27, "The free end hangs straight down")
    }

    func testSheetCatchesADroppedBall() throws {
        let backend = makeBackend()
        let sheet = try XCTUnwrap(backend.addSoftBody(sheet(n: 7, spacing: 0.1, y: 1.0)))
        advance(backend, seconds: 0.5)
        backend.didAddBody(entity: 7, descriptor: PhysicsBodyDescriptor(
            motionType: .dynamic,
            collider: PhysicsColliderDescriptor(shape: .sphere(radius: 0.06), friction: 0.4, restitution: 0.1),
            mass: 0.2,
            position: simd_float3(0, 1.6, 0)
        ))
        advance(backend, seconds: 2.0)
        let ball = try XCTUnwrap(backend.bodyState(for: 7))
        XCTAssertGreaterThan(ball.position.y, 0.5, "The sheet holds the ball up (the floor is at 0)")
        var positions: [SIMD3<Float>] = []
        backend.readSoftBodyVertices(sheet, into: &positions)
        let centre = positions[3 * 7 + 3]
        XCTAssertLessThan(centre.y, 0.98, "The sheet sags under the ball")
        XCTAssertEqual(positions[0].y, 1.0, accuracy: 1e-4, "A pinned corner stays put")
    }

    func testRemovedSoftBodyStopsExisting() throws {
        let backend = makeBackend()
        let body = try XCTUnwrap(backend.addSoftBody(sheet(n: 3, spacing: 0.1, y: 1.0)))
        advance(backend, seconds: 0.2)
        backend.removeSoftBody(body)
        advance(backend, seconds: 0.2)
        var positions: [SIMD3<Float>] = []
        XCTAssertEqual(backend.readSoftBodyVertices(body, into: &positions), 0)
        XCTAssertNil(backend.addSoftBody(JoltSoftBodyDescriptor(vertices: [.zero, .zero], inverseMasses: [0, 1], edges: [SIMD2(0, 5)])), "A bad edge index is refused")
    }
}
