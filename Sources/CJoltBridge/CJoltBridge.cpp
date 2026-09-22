//
//  CJoltBridge.cpp
//  UntoldJoltPhysics
//
//  Implementation of the C API over Jolt. One PhysicsSystem per world,
//  table-driven layer filtering (engine layer 0..31 x static/moving), and
//  fixed-capacity event buffers filled from Jolt's listeners (which run on
//  its worker threads) and drained on the caller's thread.
//

#include <Jolt/Jolt.h>

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemSingleThreaded.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Geometry/AABox.h>
#include <Jolt/Physics/Body/BodyActivationListener.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyFilter.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Character/CharacterVirtual.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayerInterfaceTable.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/ContactListener.h>
#include <Jolt/Physics/Collision/ObjectLayerPairFilterTable.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/TransformedShape.h>
#include <Jolt/Physics/Constraints/SwingTwistConstraint.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/Ragdoll/Ragdoll.h>
#include <Jolt/Physics/Collision/GroupFilterTable.h>
#include <Jolt/Physics/SoftBody/SoftBodyCreationSettings.h>
#include <Jolt/Physics/SoftBody/SoftBodyMotionProperties.h>
#include <Jolt/Physics/SoftBody/SoftBodySharedSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Skeleton/Skeleton.h>

#include <algorithm>
#include <cfloat>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "CJoltBridge.h"

using namespace JPH;

namespace {

// Object layer = engine layer (0..31) * 2 + moving bit. Broad phase: 2 layers.
constexpr uint kEngineLayers = 32;
constexpr uint kObjectLayers = kEngineLayers * 2;
constexpr BroadPhaseLayer kBPStatic(0);
constexpr BroadPhaseLayer kBPMoving(1);
constexpr uint kBPLayers = 2;
constexpr uint32_t kEventCapacity = 2048;

inline ObjectLayer objectLayer(uint32_t engineLayer, bool moving) {
    return ObjectLayer(std::min(engineLayer, kEngineLayers - 1) * 2 + (moving ? 1 : 0));
}
inline uint32_t engineLayerOf(ObjectLayer layer) { return uint32_t(layer) / 2; }

inline Vec3 v3(const float *p) { return Vec3(p[0], p[1], p[2]); }
inline Quat q4(const float *p) {
    Quat q(p[0], p[1], p[2], p[3]);
    return q.IsNormalized() ? q : (q.LengthSq() > 1e-12f ? q.Normalized() : Quat::sIdentity());
}
inline void store3(float *out, Vec3Arg v) { out[0] = v.GetX(); out[1] = v.GetY(); out[2] = v.GetZ(); }
inline void store4(float *out, QuatArg q) { out[0] = q.GetX(); out[1] = q.GetY(); out[2] = q.GetZ(); out[3] = q.GetW(); }
/// 16 floats column-major (simd_float4x4 layout), which is Mat44's own.
inline Mat44 mat44(const float *m) {
    return Mat44(Vec4(m[0], m[1], m[2], m[3]), Vec4(m[4], m[5], m[6], m[7]),
                 Vec4(m[8], m[9], m[10], m[11]), Vec4(m[12], m[13], m[14], m[15]));
}
/// A pose matrix's rotation. The columns are orthonormalised first (Gram-
/// Schmidt from X, right-handed), so a scaled or drifted game matrix — a
/// scaled entity, accumulated float error — yields the rotation it means
/// rather than the garbage GetQuaternion makes of a non-orthonormal 3x3.
inline Quat rotationOf(const Mat44 &matrix) {
    Vec3 x = matrix.GetAxisX();
    x = x.LengthSq() > 1e-12f ? x.Normalized() : Vec3::sAxisX();
    Vec3 y = matrix.GetAxisY();
    y -= x * y.Dot(x);
    y = y.LengthSq() > 1e-12f ? y.Normalized() : x.GetNormalizedPerpendicular();
    const Vec3 z = x.Cross(y);
    return Mat44(Vec4(x, 0.0f), Vec4(y, 0.0f), Vec4(z, 0.0f), Vec4(0.0f, 0.0f, 0.0f, 1.0f)).GetQuaternion().Normalized();
}
inline Quat rotationOf(const float *matrix) { return rotationOf(mat44(matrix)); }
inline void storeMat44(float *out, Mat44Arg m) {
    for (uint c = 0; c < 4; ++c) {
        const Vec4 column = m.GetColumn4(c);
        out[c * 4] = column.GetX();
        out[c * 4 + 1] = column.GetY();
        out[c * 4 + 2] = column.GetZ();
        out[c * 4 + 3] = column.GetW();
    }
}

struct BodyRecord {
    uint64_t userData;
    bool sensor;
    EMotionType motion;
    bool soft = false;
    /// A character controller's inner body: owned by the character (which
    /// removes and destroys it), listed here so contacts and rays resolve
    /// to the character's user data.
    bool inner = false;
    /// A ragdoll part: owned by the ragdoll (whose Jolt object destroys it),
    /// listed here for contacts and rays; never handed to the engine's
    /// coordinator as a changed body.
    bool ragdoll = false;
};

struct KinematicTarget {
    BodyID id;
    RVec3 position;
    Quat rotation;
};

/// Broad-phase gate: only static-vs-static is rejected here; the pair filter
/// table (which the layer matrix edits at runtime) does the fine filtering.
/// Jolt's ObjectVsBroadPhaseLayerFilterTable would snapshot the pair table at
/// construction, so it cannot follow later matrix changes.
class MovingVsStaticBroadPhaseFilter final : public ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(ObjectLayer layer, BroadPhaseLayer bpLayer) const override {
        const bool moving = (uint32_t(layer) & 1u) != 0;
        return moving || !(bpLayer == kBPStatic);
    }
};

void ensureJoltRegistered() {
    static std::once_flag once;
    std::call_once(once, [] {
        RegisterDefaultAllocator();
        Factory::sInstance = new Factory();
        RegisterTypes();
    });
}

} // namespace

/// A Jolt CharacterVirtual and what its moves need. Owned by the world it
/// was added to (released before the world's bodies on destruction).
struct ujolt_character final : public CharacterContactListener {
    ujolt_world *world = nullptr;
    Ref<CharacterVirtual> character;
    ObjectLayer layer = 0;
    CharacterVirtual::ExtendedUpdateSettings extended;
    ujolt_body_id innerBody = UJOLT_INVALID_BODY;
    bool pushedByDynamicBodies = false;
    bool ignoresDynamicBodies = false;
    /// What the last move produced: displacement over its dt.
    Vec3 effectiveVelocity = Vec3::sZero();

    // A discarded contact never becomes a constraint nor an impulse: with
    // dynamic bodies ignored, a ball flies into the inner body (a plain
    // kinematic body the world's listener reports) instead of being caught
    // and stopped by the character's own collision, which no listener sees.
    bool OnContactValidate(const CharacterVirtual *, const CharacterContact &contact) override {
        return !(ignoresDynamicBodies && contact.mMotionTypeB == EMotionType::Dynamic);
    }

    // Synchronous, on the thread running the move. With mCanPushCharacter
    // off Jolt drops the constraint's velocity (the body's own and the
    // penetration recovery), so a dynamic body neither shoves the character
    // nor nudges it out of a resting touch; the constraint plane still stops
    // the character walking through it, and the impulse on the body stays.
    void OnContactAdded(const CharacterVirtual *, const CharacterContact &contact, CharacterContactSettings &settings) override {
        settings.mCanPushCharacter = pushedByDynamicBodies || contact.mMotionTypeB != EMotionType::Dynamic;
    }
    void OnContactPersisted(const CharacterVirtual *, const CharacterContact &contact, CharacterContactSettings &settings) override {
        settings.mCanPushCharacter = pushedByDynamicBodies || contact.mMotionTypeB != EMotionType::Dynamic;
    }
};

/// A Jolt Ragdoll and what driving it needs. Owned by the world it was
/// added to (released before the world's bodies on destruction): the Ragdoll
/// destructor destroys its bodies, which must have been removed first.
struct ujolt_ragdoll {
    ujolt_world *world = nullptr;
    Ref<Skeleton> skeleton;
    Ref<RagdollSettings> settings;
    Ref<Ragdoll> ragdoll;
    std::vector<ujolt_body_id> parts;
    /// Per part, the mode its constraint's motors were last set to (the
    /// root has no constraint; its entry is never read).
    std::vector<ujolt_motor_mode> motors;
    /// The pose the kinematic parts are moved to every step while one is
    /// set: the engine runs 0..5 substeps a frame, and re-driving each
    /// substep is what a kinematic target does.
    std::vector<Mat44> kinematicPose;
    bool kinematicPending = false;
    bool active = false;
};

struct ujolt_world final : public ContactListener, public BodyActivationListener {
    ujolt_world(const ujolt_world_desc &desc)
        : bpInterface(kObjectLayers, kBPLayers),
          pairFilter(kObjectLayers),
          reportPersisted(desc.report_persisted_contacts != 0),
          maxKinematicStep(desc.max_kinematic_step),
          maxKinematicSpeed(desc.max_kinematic_speed),
          continuousCollision(desc.continuous_collision != 0),
          minContactSpeed(desc.min_contact_speed)
    {
        for (uint32_t layer = 0; layer < kEngineLayers; ++layer) {
            bpInterface.MapObjectToBroadPhaseLayer(objectLayer(layer, false), kBPStatic);
            bpInterface.MapObjectToBroadPhaseLayer(objectLayer(layer, true), kBPMoving);
        }
        applyLayerMatrix(nullptr, 0);

        temp = new TempAllocatorImpl(desc.temp_allocator_bytes ? desc.temp_allocator_bytes : 16u * 1024u * 1024u);
        int threads = desc.worker_threads;
        if (threads < 0) {
            threads = int(std::thread::hardware_concurrency()) - 1;
        }
        if (threads <= 0) {
            jobs = new JobSystemSingleThreaded(cMaxPhysicsJobs);
        } else {
            jobs = new JobSystemThreadPool(cMaxPhysicsJobs, cMaxPhysicsBarriers, threads);
        }

        const uint maxBodies = desc.max_bodies ? desc.max_bodies : 4096;
        const uint maxPairs = desc.max_body_pairs ? desc.max_body_pairs : 4096;
        const uint maxContacts = desc.max_contact_constraints ? desc.max_contact_constraints : 2048;
        system.Init(maxBodies, 0, maxPairs, maxContacts, bpInterface, objVsBpFilter, pairFilter);
        system.SetGravity(v3(desc.gravity));
        system.SetContactListener(this);
        system.SetBodyActivationListener(this);
    }

    ~ujolt_world() override {
        system.SetContactListener(nullptr);
        system.SetBodyActivationListener(nullptr);
        // Characters own their inner bodies (the CharacterVirtual destructor
        // removes and destroys them): release them before the body sweep
        // below, and while the PhysicsSystem is still alive.
        for (ujolt_character *character : characters) releaseCharacter(character, false);
        characters.clear();
        // Likewise the ragdolls: the Ragdoll destructor destroys the parts
        // and wants them out of the system first.
        for (ujolt_ragdoll *ragdoll : ragdolls) releaseRagdoll(ragdoll, false);
        ragdolls.clear();
        BodyInterface &bi = system.GetBodyInterface();
        for (auto &entry : records) {
            BodyID id(entry.first);
            bi.RemoveBody(id);
            bi.DestroyBody(id);
        }
        records.clear();
        delete jobs;
        delete temp;
    }

    // MARK: Layers

    void applyLayerMatrix(const uint32_t *masks, uint32_t count) {
        for (uint32_t a = 0; a < kEngineLayers; ++a) {
            for (uint32_t b = 0; b < kEngineLayers; ++b) {
                bool allowed;
                if (masks == nullptr || count == 0) {
                    allowed = true;
                } else {
                    const uint32_t maskA = a < count ? masks[a] : 0xFFFFFFFFu;
                    const uint32_t maskB = b < count ? masks[b] : 0xFFFFFFFFu;
                    allowed = ((maskA >> b) & 1u) != 0 || ((maskB >> a) & 1u) != 0;
                }
                // Static never collides with static; everything else per matrix.
                for (int movingA = 0; movingA < 2; ++movingA) {
                    for (int movingB = 0; movingB < 2; ++movingB) {
                        const ObjectLayer la = objectLayer(a, movingA != 0);
                        const ObjectLayer lb = objectLayer(b, movingB != 0);
                        if (allowed && (movingA || movingB)) {
                            pairFilter.EnableCollision(la, lb);
                        } else {
                            pairFilter.DisableCollision(la, lb);
                        }
                    }
                }
            }
        }
    }

    // MARK: Records (guarded: listeners read from worker threads)

    bool lookup(const BodyID &id, BodyRecord &out) const {
        std::lock_guard<std::mutex> guard(recordMutex);
        auto it = records.find(id.GetIndexAndSequenceNumber());
        if (it == records.end()) {
            // Jolt reports OnContactRemoved for a deleted body during the
            // NEXT update; the tombstone keeps its identity until the events
            // of that update have been drained, so trigger exits survive.
            it = removedRecords.find(id.GetIndexAndSequenceNumber());
            if (it == removedRecords.end()) return false;
        }
        out = it->second;
        return true;
    }

    // MARK: Characters (frame thread)

    /// Forgets the character's inner body (as a tombstone when `keepTombstone`,
    /// so an in-flight OnContactRemoved still resolves it) and deletes the
    /// character, whose destructor removes and destroys the inner body.
    void releaseCharacter(ujolt_character *character, bool keepTombstone) {
        if (character->innerBody != UJOLT_INVALID_BODY) {
            std::lock_guard<std::mutex> guard(recordMutex);
            auto it = records.find(character->innerBody);
            if (it != records.end()) {
                if (keepTombstone) removedRecords[character->innerBody] = it->second;
                records.erase(it);
            }
        }
        delete character;
    }

    // MARK: Ragdolls (frame thread)

    /// Takes the ragdoll out of the system if it is in, forgets its parts
    /// (as tombstones when `keepTombstone`, so an in-flight OnContactRemoved
    /// still resolves them) and deletes it, which drops the Ref whose last
    /// holder destroys the bodies.
    void releaseRagdoll(ujolt_ragdoll *ragdoll, bool keepTombstone) {
        if (ragdoll->active) {
            ragdoll->ragdoll->RemoveFromPhysicsSystem();
            ragdoll->active = false;
        }
        {
            std::lock_guard<std::mutex> guard(recordMutex);
            for (ujolt_body_id part : ragdoll->parts) {
                auto it = records.find(part);
                if (it == records.end()) continue;
                if (keepTombstone) removedRecords[part] = it->second;
                records.erase(it);
            }
        }
        delete ragdoll;
    }

    // MARK: ContactListener (Jolt worker threads)

    ValidateResult OnContactValidate(const Body &, const Body &, RVec3Arg, const CollideShapeResult &) override {
        return ValidateResult::AcceptAllContactsForThisBodyPair;
    }

    void OnContactAdded(const Body &b1, const Body &b2, const ContactManifold &manifold, ContactSettings &) override {
        pushContact(UJOLT_CONTACT_ADDED, b1, b2, manifold);
    }

    void OnContactPersisted(const Body &b1, const Body &b2, const ContactManifold &manifold, ContactSettings &) override {
        if (reportPersisted) pushContact(UJOLT_CONTACT_PERSISTED, b1, b2, manifold);
    }

    static uint64_t pairKey(const BodyID &a, const BodyID &b) {
        const uint32_t x = a.GetIndexAndSequenceNumber();
        const uint32_t y = b.GetIndexAndSequenceNumber();
        return x < y ? (uint64_t(x) << 32) | y : (uint64_t(y) << 32) | x;
    }

    void OnContactRemoved(const SubShapeIDPair &pair) override {
        BodyRecord r1, r2;
        if (!lookup(pair.GetBody1ID(), r1) || !lookup(pair.GetBody2ID(), r2)) return;
        const bool anySensor = r1.sensor || r2.sensor;
        if (anySensor) {
            // Triggers report dynamic occupants only (engine trigger semantics).
            const BodyRecord &other = r1.sensor ? r2 : r1;
            if (other.motion != EMotionType::Dynamic) return;
        }
        // Same A/B rule as pushContact, from the records.
        const bool aIsBody2 = r2.motion == EMotionType::Dynamic || r1.motion != EMotionType::Dynamic;
        const BodyRecord &ra = aIsBody2 ? r2 : r1;
        const BodyRecord &rb = aIsBody2 ? r1 : r2;
        std::lock_guard<std::mutex> guard(eventMutex);
        if (!anySensor && reportedPairs.erase(pairKey(pair.GetBody1ID(), pair.GetBody2ID())) == 0) {
            return; // its ADDED was filtered as a resting contact
        }
        ujolt_contact_event event{};
        event.phase = UJOLT_CONTACT_REMOVED;
        event.user_a = ra.userData;
        event.user_b = rb.userData;
        event.sensor_a = ra.sensor ? 1 : 0;
        event.sensor_b = rb.sensor ? 1 : 0;
        if (contacts.size() < kEventCapacity) contacts.push_back(event); else ++droppedContacts;
    }

    void pushContact(ujolt_contact_phase phase, const Body &b1, const Body &b2, const ContactManifold &manifold) {
        // Convention (matches the engine's reference backend): A is the
        // dynamic body — body 2 when both or neither are — and the normal
        // points out of B into A. Jolt's manifold normal points from body 1
        // toward body 2.
        const bool anySensor = b1.IsSensor() || b2.IsSensor();
        if (anySensor) {
            // Kinematic sensors also see kinematic bodies in Jolt; the engine's
            // triggers report dynamic occupants only.
            const Body &other = b1.IsSensor() ? b2 : b1;
            if (!other.IsDynamic()) return;
        }
        const bool aIsBody2 = b2.IsDynamic() || !b1.IsDynamic();
        const Body &a = aIsBody2 ? b2 : b1;
        const Body &b = aIsBody2 ? b1 : b2;
        ujolt_contact_event event{};
        event.phase = phase;
        event.user_a = a.GetUserData();
        event.user_b = b.GetUserData();
        event.sensor_a = a.IsSensor() ? 1 : 0;
        event.sensor_b = b.IsSensor() ? 1 : 0;
        if (!manifold.mRelativeContactPointsOn1.empty()) {
            store3(event.position, Vec3(manifold.GetWorldSpaceContactPointOn1(0)));
        }
        store3(event.normal, aIsBody2 ? manifold.mWorldSpaceNormal : -manifold.mWorldSpaceNormal);

        // Estimated normal impulse: closing speed along the normal times the
        // reduced mass. Jolt does not expose the solved impulse in this
        // callback; this is what a sound/FX layer needs.
        const float invMass1 = b1.IsDynamic() ? b1.GetMotionProperties()->GetInverseMass() : 0.0f;
        const float invMass2 = b2.IsDynamic() ? b2.GetMotionProperties()->GetInverseMass() : 0.0f;
        const float invSum = invMass1 + invMass2;
        const float reducedMass = invSum > 0.0f ? 1.0f / invSum : 0.0f;
        const Vec3 relative = b2.GetLinearVelocity() - b1.GetLinearVelocity();
        const float closing = std::max(0.0f, -relative.Dot(manifold.mWorldSpaceNormal));
        event.impulse = closing * reducedMass;

        std::lock_guard<std::mutex> guard(eventMutex);
        if (!anySensor && phase == UJOLT_CONTACT_ADDED) {
            // A contact re-established at rest (after a sleep, or a geometry
            // rebuild) is not an impact.
            if (minContactSpeed > 0.0f && closing < minContactSpeed) return;
            reportedPairs.insert(pairKey(b1.GetID(), b2.GetID()));
        }
        if (contacts.size() < kEventCapacity) contacts.push_back(event); else ++droppedContacts;
    }

    // MARK: BodyActivationListener (Jolt worker threads)

    void OnBodyActivated(const BodyID &, uint64 userData) override {
        std::lock_guard<std::mutex> guard(eventMutex);
        if (activations.size() < kEventCapacity) activations.push_back({userData, 1}); else ++droppedActivations;
    }

    void OnBodyDeactivated(const BodyID &id, uint64 userData) override {
        std::lock_guard<std::mutex> guard(eventMutex);
        if (activations.size() < kEventCapacity) activations.push_back({userData, 0}); else ++droppedActivations;
        deactivatedThisStep.push_back(id);
    }

    // MARK: State

    TempAllocatorImpl *temp = nullptr;
    JobSystem *jobs = nullptr;
    BroadPhaseLayerInterfaceTable bpInterface;
    ObjectLayerPairFilterTable pairFilter;
    MovingVsStaticBroadPhaseFilter objVsBpFilter;
    PhysicsSystem system;
    bool reportPersisted;
    float maxKinematicStep;
    float maxKinematicSpeed;
    bool continuousCollision;
    float minContactSpeed;
    bool broadPhaseDirty = false;

    mutable std::mutex recordMutex;
    std::unordered_map<uint32_t, BodyRecord> records;
    std::unordered_map<uint32_t, BodyRecord> removedRecords;
    std::vector<uint32_t> removedSinceLastStep;

    std::vector<KinematicTarget> kinematicTargets;
    std::unordered_set<ujolt_character *> characters;
    std::unordered_set<ujolt_ragdoll *> ragdolls;
    /// Jolt's collision group per ragdoll (parent/child pairs are filtered
    /// within a group): unique per ragdoll, so two never share one.
    uint32_t nextRagdollGroup = 1;

    std::mutex eventMutex;
    std::unordered_set<uint64_t> reportedPairs;
    std::vector<ujolt_contact_event> contacts;
    std::vector<ujolt_activation_event> activations;
    std::vector<BodyID> deactivatedThisStep;
    uint32_t droppedContacts = 0;
    uint32_t droppedActivations = 0;
};

namespace {

/// The bare shape at the body origin; the callers place it.
RefConst<Shape> makePrimitiveShape(ujolt_shape_type kind, float radius, float halfHeight, const float *halfExtents,
                                   const float *hullPoints, uint32_t hullPointCount) {
    RefConst<Shape> shape;
    switch (kind) {
    case UJOLT_SHAPE_SPHERE:
        shape = new SphereShape(std::max(radius, 1e-4f));
        break;
    case UJOLT_SHAPE_BOX: {
        Vec3 half = Vec3::sMax(v3(halfExtents), Vec3::sReplicate(1e-4f));
        shape = new BoxShape(half);
        break;
    }
    case UJOLT_SHAPE_CAPSULE:
        shape = new CapsuleShape(std::max(halfHeight, 1e-4f), std::max(radius, 1e-4f));
        break;
    case UJOLT_SHAPE_CYLINDER: {
        const float h = std::max(halfHeight, 1e-4f);
        const float r = std::max(radius, 1e-4f);
        shape = new CylinderShape(h, r, std::min(cDefaultConvexRadius, std::min(h, r)));
        break;
    }
    case UJOLT_SHAPE_CONVEX_HULL: {
        if (hullPoints == nullptr || hullPointCount < 4) return nullptr;
        std::vector<Vec3> points;
        points.reserve(hullPointCount);
        Vec3 lo = Vec3::sReplicate(FLT_MAX), hi = Vec3::sReplicate(-FLT_MAX);
        for (uint32_t i = 0; i < hullPointCount; ++i) {
            const Vec3 p = v3(hullPoints + i * 3);
            points.push_back(p);
            lo = Vec3::sMin(lo, p);
            hi = Vec3::sMax(hi, p);
        }
        // Jolt shrinks a hull by its convex radius and collides the rounded
        // result; the 5 cm default is a third of a bowling pin's belly, so
        // small hulls would collide well outside their mesh (and lose their
        // necks). Scale the radius with the hull's smallest extent.
        const float smallestExtent = (hi - lo).ReduceMin();
        const float convexRadius = std::min(cDefaultConvexRadius, std::max(0.0f, smallestExtent * 0.1f));
        ConvexHullShapeSettings settings(points.data(), int(points.size()), convexRadius);
        Shape::ShapeResult result = settings.Create();
        if (result.HasError()) return nullptr;
        shape = result.Get();
        break;
    }
    }
    return shape;
}

RefConst<Shape> makeShape(const ujolt_body_desc &desc) {
    RefConst<Shape> shape = makePrimitiveShape(desc.shape, desc.radius, desc.half_height, desc.half_extents,
                                               desc.hull_points, desc.hull_point_count);
    if (shape == nullptr) return nullptr;

    const Vec3 offset = v3(desc.local_offset);
    if (offset.LengthSq() > 1e-12f) {
        shape = new RotatedTranslatedShape(offset, Quat::sIdentity(), shape);
    }
    return shape;
}

EMotionType motionType(ujolt_motion_type motion) {
    switch (motion) {
    case UJOLT_MOTION_STATIC: return EMotionType::Static;
    case UJOLT_MOTION_KINEMATIC: return EMotionType::Kinematic;
    default: return EMotionType::Dynamic;
    }
}

class LayerMaskFilter final : public ObjectLayerFilter {
public:
    explicit LayerMaskFilter(uint32_t mask) : mMask(mask) {}
    bool ShouldCollide(ObjectLayer layer) const override {
        return ((mMask >> engineLayerOf(layer)) & 1u) != 0;
    }
private:
    uint32_t mMask;
};

class ExcludedUserDataFilter final : public BodyFilter {
public:
    ExcludedUserDataFilter(const ujolt_world &world, const uint64_t *excluded, uint32_t count)
        : mWorld(world), mExcluded(excluded), mCount(count) {}
    /// Trigger volumes are not solid: rays pass through them.
    bool ShouldCollideLocked(const Body &body) const override { return !body.IsSensor(); }
    bool ShouldCollide(const BodyID &id) const override {
        if (mCount == 0) return true;
        BodyRecord record;
        if (!mWorld.lookup(id, record)) return true;
        for (uint32_t i = 0; i < mCount; ++i) {
            if (mExcluded[i] == record.userData) return false;
        }
        return true;
    }
private:
    const ujolt_world &mWorld;
    const uint64_t *mExcluded;
    uint32_t mCount;
};

} // namespace

// MARK: - C API

extern "C" {

ujolt_world *ujolt_world_create(const ujolt_world_desc *desc) {
    ensureJoltRegistered();
    ujolt_world_desc defaults{};
    defaults.gravity[1] = -9.81f;
    defaults.worker_threads = -1;
    return new ujolt_world(desc ? *desc : defaults);
}

void ujolt_world_destroy(ujolt_world *world) {
    delete world;
}

void ujolt_world_set_gravity(ujolt_world *world, const float gravity[3]) {
    world->system.SetGravity(v3(gravity));
}

void ujolt_world_set_layer_matrix(ujolt_world *world, const uint32_t *layer_masks, uint32_t count) {
    world->applyLayerMatrix(layer_masks, count);
}

ujolt_body_id ujolt_world_add_body(ujolt_world *world, const ujolt_body_desc *desc) {
    RefConst<Shape> shape = makeShape(*desc);
    if (shape == nullptr) return UJOLT_INVALID_BODY;

    // A static sensor only detects ACTIVE bodies and reports the contact as
    // removed when its occupant falls asleep — the engine's trigger semantics
    // are occupancy, so sensors are made kinematic and kept awake. (Kinematic
    // sensors also see kinematic bodies; the listener filters those out.)
    const bool sensor = desc->is_sensor != 0;
    const EMotionType motion = sensor && desc->motion == UJOLT_MOTION_STATIC
        ? EMotionType::Kinematic
        : motionType(desc->motion);
    const bool moving = motion != EMotionType::Static;
    BodyCreationSettings settings(shape, RVec3(v3(desc->position)), q4(desc->rotation), motion,
                                  objectLayer(desc->layer, moving));
    if (sensor) settings.mAllowSleeping = false;
    settings.mUserData = desc->user_data;
    settings.mFriction = desc->friction;
    settings.mRestitution = desc->restitution;
    settings.mIsSensor = sensor;
    settings.mGravityFactor = desc->gravity_factor;
    settings.mLinearVelocity = v3(desc->linear_velocity);
    settings.mAngularVelocity = v3(desc->angular_velocity);
    if (motion == EMotionType::Dynamic && desc->mass > 0.0f) {
        settings.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
        settings.mMassPropertiesOverride.mMass = desc->mass;
    }
    if (motion == EMotionType::Dynamic && world->continuousCollision) {
        settings.mMotionQuality = EMotionQuality::LinearCast;
    }

    BodyInterface &bi = world->system.GetBodyInterface();
    const BodyID id = bi.CreateAndAddBody(settings, moving ? EActivation::Activate : EActivation::DontActivate);
    if (id.IsInvalid()) return UJOLT_INVALID_BODY;

    {
        std::lock_guard<std::mutex> guard(world->recordMutex);
        world->records[id.GetIndexAndSequenceNumber()] = BodyRecord{desc->user_data, sensor, motionType(desc->motion)};
    }
    world->broadPhaseDirty = true;
    return id.GetIndexAndSequenceNumber();
}

ujolt_body_id ujolt_world_add_soft_body(ujolt_world *world, const ujolt_soft_body_desc *desc) {
    if (desc->vertex_count == 0 || desc->vertices == nullptr) return UJOLT_INVALID_BODY;
    Ref<SoftBodySharedSettings> shared = new SoftBodySharedSettings;
    shared->mVertices.reserve(desc->vertex_count);
    for (uint32_t i = 0; i < desc->vertex_count; ++i) {
        const float *p = desc->vertices + i * 3;
        const float invMass = desc->inv_masses ? desc->inv_masses[i] : 1.0f;
        shared->mVertices.push_back(SoftBodySharedSettings::Vertex(Float3(p[0], p[1], p[2]), Float3(0.0f, 0.0f, 0.0f), invMass));
    }
    for (uint32_t i = 0; i < desc->edge_count; ++i) {
        const uint32_t a = desc->edges[i * 2], b = desc->edges[i * 2 + 1];
        if (a >= desc->vertex_count || b >= desc->vertex_count || a == b) return UJOLT_INVALID_BODY;
        const float compliance = desc->edge_compliances ? desc->edge_compliances[i] : desc->compliance;
        shared->mEdgeConstraints.push_back(SoftBodySharedSettings::Edge(a, b, compliance));
    }
    for (uint32_t i = 0; i < desc->face_count; ++i) {
        const uint32_t *f = desc->faces + i * 3;
        if (f[0] >= desc->vertex_count || f[1] >= desc->vertex_count || f[2] >= desc->vertex_count) return UJOLT_INVALID_BODY;
        shared->mFaces.push_back(SoftBodySharedSettings::Face(f[0], f[1], f[2]));
    }
    shared->CalculateEdgeLengths();
    shared->Optimize();

    SoftBodyCreationSettings settings(shared, RVec3(v3(desc->position)), Quat::sIdentity(), objectLayer(desc->layer, true));
    settings.mUserData = desc->user_data;
    if (desc->iterations > 0) settings.mNumIterations = desc->iterations;
    settings.mLinearDamping = desc->linear_damping;
    settings.mVertexRadius = desc->vertex_radius;
    settings.mFriction = desc->friction;
    settings.mRestitution = desc->restitution;
    settings.mGravityFactor = desc->gravity_factor;
    // Hangs from its pinned vertices: the body's frame stays where it was made.
    settings.mUpdatePosition = false;
    settings.mMakeRotationIdentity = true;

    BodyInterface &bi = world->system.GetBodyInterface();
    const BodyID id = bi.CreateAndAddSoftBody(settings, EActivation::Activate);
    if (id.IsInvalid()) return UJOLT_INVALID_BODY;
    {
        std::lock_guard<std::mutex> guard(world->recordMutex);
        BodyRecord record{desc->user_data, false, EMotionType::Dynamic};
        record.soft = true;
        world->records[id.GetIndexAndSequenceNumber()] = record;
    }
    world->broadPhaseDirty = true;
    return id.GetIndexAndSequenceNumber();
}

uint32_t ujolt_world_soft_body_vertex_count(ujolt_world *world, ujolt_body_id body) {
    BodyLockRead lock(world->system.GetBodyLockInterface(), BodyID(body));
    if (!lock.Succeeded() || !lock.GetBody().IsSoftBody()) return 0;
    const auto *mp = static_cast<const SoftBodyMotionProperties *>(lock.GetBody().GetMotionProperties());
    return uint32_t(mp->GetVertices().size());
}

uint32_t ujolt_world_read_soft_body_vertices(ujolt_world *world, ujolt_body_id body, float *positions, uint32_t capacity) {
    BodyLockRead lock(world->system.GetBodyLockInterface(), BodyID(body));
    if (!lock.Succeeded() || !lock.GetBody().IsSoftBody()) return 0;
    const Body &b = lock.GetBody();
    const auto *mp = static_cast<const SoftBodyMotionProperties *>(b.GetMotionProperties());
    // Vertex positions are relative to the body's centre of mass.
    const RMat44 com = b.GetCenterOfMassTransform();
    const auto &vertices = mp->GetVertices();
    const uint32_t count = std::min<uint32_t>(capacity, uint32_t(vertices.size()));
    for (uint32_t i = 0; i < count; ++i) {
        const RVec3 w = com * vertices[i].mPosition;
        positions[i * 3] = float(w.GetX());
        positions[i * 3 + 1] = float(w.GetY());
        positions[i * 3 + 2] = float(w.GetZ());
    }
    return count;
}

void ujolt_world_remove_body(ujolt_world *world, ujolt_body_id body) {
    const BodyID id(body);
    {
        std::lock_guard<std::mutex> guard(world->recordMutex);
        auto it = world->records.find(body);
        if (it == world->records.end()) return;
        if (it->second.inner) return; // owned by its character; goes with it
        if (it->second.ragdoll) return; // owned by its ragdoll; goes with it
        world->removedRecords[body] = it->second;
        world->records.erase(it);
    }
    BodyInterface &bi = world->system.GetBodyInterface();
    if (bi.IsAdded(id)) {
        // Jolt does not wake bodies resting on a body that goes away: they
        // would hover asleep. Wake whatever overlaps it first.
        AABox bounds = bi.GetTransformedShape(id).GetWorldSpaceBounds();
        bounds.ExpandBy(Vec3::sReplicate(0.05f));
        bi.ActivateBodiesInAABox(bounds, BroadPhaseLayerFilter(), ObjectLayerFilter());
    }
    bi.RemoveBody(id);
    bi.DestroyBody(id);
    auto &targets = world->kinematicTargets;
    targets.erase(std::remove_if(targets.begin(), targets.end(),
                                 [&](const KinematicTarget &t) { return t.id == id; }),
                  targets.end());
}

uint32_t ujolt_world_body_count(const ujolt_world *world) {
    std::lock_guard<std::mutex> guard(world->recordMutex);
    return uint32_t(world->records.size());
}

uint64_t ujolt_world_get_user_data(const ujolt_world *world, ujolt_body_id body) {
    BodyRecord record;
    return world->lookup(BodyID(body), record) ? record.userData : UJOLT_NO_ENTITY;
}

void ujolt_world_set_kinematic_target(ujolt_world *world, ujolt_body_id body, const float position[3], const float rotation[4]) {
    const BodyID id(body);
    for (auto &target : world->kinematicTargets) {
        if (target.id == id) {
            target.position = RVec3(v3(position));
            target.rotation = q4(rotation);
            return;
        }
    }
    world->kinematicTargets.push_back({id, RVec3(v3(position)), q4(rotation)});
}

void ujolt_world_set_transform(ujolt_world *world, ujolt_body_id body, const float position[3], const float rotation[4]) {
    world->system.GetBodyInterface().SetPositionAndRotation(BodyID(body), RVec3(v3(position)), q4(rotation), EActivation::Activate);
}

void ujolt_world_set_velocity(ujolt_world *world, ujolt_body_id body, const float linear[3], const float angular[3]) {
    world->system.GetBodyInterface().SetLinearAndAngularVelocity(BodyID(body), v3(linear), v3(angular));
}

void ujolt_world_get_transform(const ujolt_world *world, ujolt_body_id body, float position[3], float rotation[4]) {
    RVec3 p;
    Quat r;
    world->system.GetBodyInterface().GetPositionAndRotation(BodyID(body), p, r);
    store3(position, Vec3(p));
    store4(rotation, r);
}

void ujolt_world_get_velocity(const ujolt_world *world, ujolt_body_id body, float linear[3], float angular[3]) {
    Vec3 l, a;
    world->system.GetBodyInterface().GetLinearAndAngularVelocity(BodyID(body), l, a);
    store3(linear, l);
    store3(angular, a);
}

int32_t ujolt_world_body_is_active(const ujolt_world *world, ujolt_body_id body) {
    return world->system.GetBodyInterface().IsActive(BodyID(body)) ? 1 : 0;
}

void ujolt_world_activate_in_box(ujolt_world *world, const float min[3], const float max[3]) {
    world->system.GetBodyInterface().ActivateBodiesInAABox(
        AABox(v3(min), v3(max)), BroadPhaseLayerFilter(), ObjectLayerFilter());
}

void ujolt_world_step(ujolt_world *world, float dt, int32_t collision_steps) {
    if (dt <= 0.0f) return;
    BodyInterface &bi = world->system.GetBodyInterface();
    for (const KinematicTarget &target : world->kinematicTargets) {
        if (!bi.IsAdded(target.id)) continue;
        // The engine keys the write batch on the component's CURRENT motion
        // type; the Jolt body keeps the one it was created with.
        if (bi.GetMotionType(target.id) != EMotionType::Kinematic) continue;
        if (world->maxKinematicStep > 0.0f) {
            RVec3 current;
            Quat currentRotation;
            bi.GetPositionAndRotation(target.id, current, currentRotation);
            if (float(Vec3(target.position - current).Length()) > world->maxKinematicStep) {
                // A jump, not a motion: teleport without implied velocity.
                bi.SetPositionAndRotation(target.id, target.position, target.rotation, EActivation::Activate);
                bi.SetLinearAndAngularVelocity(target.id, Vec3::sZero(), Vec3::sZero());
                continue;
            }
        }
        bi.ActivateBody(target.id);
        bi.MoveKinematic(target.id, target.position, target.rotation, dt);
        if (world->maxKinematicSpeed > 0.0f) {
            // A glitch or a stalled frame must not become impact velocity:
            // the body approaches the target at the cap over a few substeps.
            const Vec3 velocity = bi.GetLinearVelocity(target.id);
            const float speed = velocity.Length();
            if (speed > world->maxKinematicSpeed) {
                bi.SetLinearVelocity(target.id, velocity * (world->maxKinematicSpeed / speed));
            }
        }
    }
    world->kinematicTargets.clear();

    // A ragdoll's kinematic parts follow its kinematic pose the same way,
    // with the same guards, and keep following it every substep until a
    // new one replaces it.
    for (ujolt_ragdoll *ragdoll : world->ragdolls) {
        if (!ragdoll->kinematicPending || !ragdoll->active) continue;
        for (size_t i = 0; i < ragdoll->parts.size(); ++i) {
            const BodyID id(ragdoll->parts[i]);
            if (bi.GetMotionType(id) != EMotionType::Kinematic) continue;
            const Mat44 &target = ragdoll->kinematicPose[i];
            const RVec3 position(target.GetTranslation());
            const Quat rotation = rotationOf(target);
            if (world->maxKinematicStep > 0.0f) {
                RVec3 current;
                Quat currentRotation;
                bi.GetPositionAndRotation(id, current, currentRotation);
                if (float(Vec3(position - current).Length()) > world->maxKinematicStep) {
                    bi.SetPositionAndRotation(id, position, rotation, EActivation::Activate);
                    bi.SetLinearAndAngularVelocity(id, Vec3::sZero(), Vec3::sZero());
                    continue;
                }
            }
            bi.ActivateBody(id);
            bi.MoveKinematic(id, position, rotation, dt);
            if (world->maxKinematicSpeed > 0.0f) {
                const Vec3 velocity = bi.GetLinearVelocity(id);
                const float speed = velocity.Length();
                if (speed > world->maxKinematicSpeed) {
                    bi.SetLinearVelocity(id, velocity * (world->maxKinematicSpeed / speed));
                }
            }
        }
    }

    if (world->broadPhaseDirty) {
        world->system.OptimizeBroadPhase();
        world->broadPhaseDirty = false;
    }
    {
        std::lock_guard<std::mutex> guard(world->eventMutex);
        world->deactivatedThisStep.clear();
    }
    world->system.Update(dt, std::max<int32_t>(collision_steps, 1), world->temp, world->jobs);

    // Tombstones of bodies removed before this update have now had their
    // OnContactRemoved callbacks; they can go once this step's events are
    // drained (the drain clears them).
    std::lock_guard<std::mutex> guard(world->recordMutex);
    world->removedSinceLastStep.clear();
    for (auto &entry : world->removedRecords) world->removedSinceLastStep.push_back(entry.first);
}

uint32_t ujolt_world_changed_bodies(ujolt_world *world, ujolt_body_id *ids, uint32_t capacity) {
    BodyIDVector active;
    world->system.GetActiveBodies(EBodyType::RigidBody, active);
    std::vector<BodyID> asleep;
    {
        std::lock_guard<std::mutex> guard(world->eventMutex);
        asleep = world->deactivatedThisStep;
    }
    uint32_t written = 0;
    auto emit = [&](const BodyID &id) {
        if (written >= capacity) return;
        BodyRecord record;
        if (!world->lookup(id, record) || record.motion != EMotionType::Dynamic || record.soft || record.ragdoll) return;
        ids[written++] = id.GetIndexAndSequenceNumber();
    };
    // Jolt's active list has no duplicates; only the (small) just-asleep
    // list must be checked against it.
    std::unordered_set<uint32_t> activeIDs;
    activeIDs.reserve(active.size());
    for (const BodyID &id : active) {
        activeIDs.insert(id.GetIndexAndSequenceNumber());
        emit(id);
    }
    for (const BodyID &id : asleep) {
        if (activeIDs.count(id.GetIndexAndSequenceNumber()) == 0) emit(id);
    }
    return written;
}

uint32_t ujolt_world_drain_contacts(ujolt_world *world, ujolt_contact_event *out, uint32_t capacity, uint32_t *dropped) {
    std::lock_guard<std::mutex> guard(world->eventMutex);
    const uint32_t count = std::min<uint32_t>(capacity, uint32_t(world->contacts.size()));
    for (uint32_t i = 0; i < count; ++i) out[i] = world->contacts[i];
    if (dropped) *dropped = world->droppedContacts + uint32_t(world->contacts.size() - count);
    world->contacts.clear();
    world->droppedContacts = 0;
    {
        std::lock_guard<std::mutex> recordGuard(world->recordMutex);
        for (uint32_t body : world->removedSinceLastStep) world->removedRecords.erase(body);
        world->removedSinceLastStep.clear();
    }
    return count;
}

uint32_t ujolt_world_drain_activations(ujolt_world *world, ujolt_activation_event *out, uint32_t capacity, uint32_t *dropped) {
    std::lock_guard<std::mutex> guard(world->eventMutex);
    const uint32_t count = std::min<uint32_t>(capacity, uint32_t(world->activations.size()));
    for (uint32_t i = 0; i < count; ++i) out[i] = world->activations[i];
    if (dropped) *dropped = world->droppedActivations + uint32_t(world->activations.size() - count);
    world->activations.clear();
    world->droppedActivations = 0;
    return count;
}

int32_t ujolt_world_cast_ray(const ujolt_world *world, const float origin[3], const float direction[3], float max_distance,
                             uint32_t layer_mask, const uint64_t *excluded_user_data, uint32_t excluded_count,
                             ujolt_ray_hit *out_hit) {
    Vec3 dir = v3(direction);
    if (dir.LengthSq() < 1e-12f || max_distance <= 0.0f) return 0;
    dir = dir.Normalized();
    const RRayCast ray(RVec3(v3(origin)), dir * max_distance);
    RayCastResult hit;
    LayerMaskFilter layerFilter(layer_mask);
    ExcludedUserDataFilter bodyFilter(*world, excluded_user_data, excluded_count);
    if (!world->system.GetNarrowPhaseQuery().CastRay(ray, hit, BroadPhaseLayerFilter(), layerFilter, bodyFilter)) {
        return 0;
    }
    const RVec3 point = ray.GetPointOnRay(hit.mFraction);
    Vec3 normal = -dir;
    {
        BodyLockRead lock(world->system.GetBodyLockInterface(), hit.mBodyID);
        if (lock.Succeeded()) {
            normal = lock.GetBody().GetWorldSpaceSurfaceNormal(hit.mSubShapeID2, point);
        }
    }
    BodyRecord record{UJOLT_NO_ENTITY, false, EMotionType::Static};
    world->lookup(hit.mBodyID, record);
    out_hit->user_data = record.userData;
    store3(out_hit->position, Vec3(point));
    store3(out_hit->normal, normal);
    out_hit->distance = hit.mFraction * max_distance;
    return 1;
}

} // extern "C"

// MARK: - Character controller

namespace {

/// A capsule or cylinder standing on the origin (CharacterVirtual wants the
/// base of the shape at (0, 0, 0)), `fraction` of the requested size but
/// sharing its centre, so an inner body sits inside the outer shape.
RefConst<Shape> characterShape(ujolt_character_shape kind, float radius, float height, float fraction) {
    const float r = std::max(radius * fraction, 1e-3f);
    const float h = std::max(height * fraction, 2.0f * r + 1e-3f);
    RefConst<Shape> shape;
    if (kind == UJOLT_CHARACTER_CYLINDER) {
        const float halfHeight = h * 0.5f;
        shape = new CylinderShape(halfHeight, r, std::min(cDefaultConvexRadius, std::min(halfHeight, r)));
    } else {
        shape = new CapsuleShape(h * 0.5f - r, r);
    }
    return new RotatedTranslatedShape(Vec3(0.0f, height * 0.5f, 0.0f), Quat::sIdentity(), shape);
}

/// Fields Jolt cannot take at zero fall back to the default at or below zero.
inline float orDefault(float value, float fallback) { return value > 0.0f ? value : fallback; }
/// Fields where zero is a meaningful choice (no mass, no push, every contact
/// a wall) fall back only when negative.
inline float orDefaultIfNegative(float value, float fallback) { return value >= 0.0f ? value : fallback; }

} // namespace

extern "C" {

ujolt_character *ujolt_world_add_character(ujolt_world *world, const ujolt_character_desc *desc) {
    if (desc == nullptr || desc->radius <= 0.0f || desc->height <= 0.0f) return nullptr;

    Ref<CharacterVirtualSettings> settings = new CharacterVirtualSettings();
    settings->mShape = characterShape(desc->shape, desc->radius, desc->height, 1.0f);
    settings->mMass = orDefaultIfNegative(desc->mass, 70.0f);
    settings->mMaxStrength = orDefaultIfNegative(desc->max_strength, 100.0f);
    settings->mCharacterPadding = orDefault(desc->padding, 0.02f);
    settings->mPredictiveContactDistance = orDefault(desc->predictive_contact_distance, 0.1f);
    settings->mMaxSlopeAngle = DegreesToRadians(orDefaultIfNegative(desc->max_slope_degrees, 50.0f));
    settings->mPenetrationRecoverySpeed = orDefault(desc->penetration_recovery_speed, 1.0f);
    // Only the lower part of the shape can rest on something: a wall touched
    // at hip height is a wall, not ground.
    settings->mSupportingVolume = Plane(Vec3::sAxisY(), -desc->radius);
    settings->mBackFaceMode = EBackFaceMode::CollideWithBackFaces;
    // Only voids internal edges within ONE body (a mesh); the environment
    // is separate convex boxes, where it would cost and change nothing.
    settings->mEnhancedInternalEdgeRemoval = false;
    const ObjectLayer layer = objectLayer(desc->layer, true);
    if (desc->inner_body != 0) {
        settings->mInnerBodyShape = characterShape(desc->shape, desc->radius, desc->height,
                                                   orDefault(desc->inner_body_fraction, 0.9f));
        settings->mInnerBodyLayer = layer;
    }

    ujolt_character *character = new ujolt_character();
    character->world = world;
    character->layer = layer;
    character->extended.mStickToFloorStepDown = Vec3(0.0f, -std::max(desc->stick_to_floor_step_down, 0.0f), 0.0f);
    character->extended.mWalkStairsStepUp = Vec3(0.0f, std::max(desc->walk_stairs_step_up, 0.0f), 0.0f);
    character->pushedByDynamicBodies = desc->pushed_by_dynamic_bodies != 0;
    character->ignoresDynamicBodies = desc->ignores_dynamic_bodies != 0;
    character->character = new CharacterVirtual(settings, RVec3(v3(desc->position)), q4(desc->rotation),
                                                desc->user_data, &world->system);
    character->character->SetListener(character);
    const BodyID inner = character->character->GetInnerBodyID();
    if (!inner.IsInvalid()) {
        character->innerBody = inner.GetIndexAndSequenceNumber();
        std::lock_guard<std::mutex> guard(world->recordMutex);
        BodyRecord record{desc->user_data, false, EMotionType::Kinematic};
        record.inner = true;
        world->records[character->innerBody] = record;
    }
    world->broadPhaseDirty = true;
    world->characters.insert(character);
    return character;
}

void ujolt_world_remove_character(ujolt_world *world, ujolt_character *character) {
    if (character == nullptr || world->characters.erase(character) == 0) return;
    world->releaseCharacter(character, true);
}

void ujolt_character_move(ujolt_character *character, const float velocity[3], float dt, const float gravity[3]) {
    if (dt <= 0.0f) return;
    ujolt_world *world = character->world;
    const RVec3 before = character->character->GetPosition();
    character->character->SetLinearVelocity(v3(velocity));
    character->character->ExtendedUpdate(dt, v3(gravity), character->extended,
                                         world->system.GetDefaultBroadPhaseLayerFilter(character->layer),
                                         world->system.GetDefaultLayerFilter(character->layer),
                                         BodyFilter(), ShapeFilter(), *world->temp);
    // Jolt keeps the velocity it was given; what the move produced is the
    // displacement (stopped at a wall it is zero, along one it is the slide).
    character->effectiveVelocity = Vec3(character->character->GetPosition() - before) / dt;
}

void ujolt_character_set_position(ujolt_character *character, const float position[3]) {
    ujolt_world *world = character->world;
    character->character->SetPosition(RVec3(v3(position)));
    character->character->RefreshContacts(world->system.GetDefaultBroadPhaseLayerFilter(character->layer),
                                          world->system.GetDefaultLayerFilter(character->layer),
                                          BodyFilter(), ShapeFilter(), *world->temp);
}

void ujolt_character_set_rotation(ujolt_character *character, const float rotation[4]) {
    character->character->SetRotation(q4(rotation));
}

void ujolt_character_get_position(const ujolt_character *character, float position[3]) {
    store3(position, Vec3(character->character->GetPosition()));
}

void ujolt_character_get_velocity(const ujolt_character *character, float velocity[3]) {
    store3(velocity, character->effectiveVelocity);
}

int32_t ujolt_character_ground_state(const ujolt_character *character) {
    switch (character->character->GetGroundState()) {
    case CharacterBase::EGroundState::OnGround: return UJOLT_GROUND_ON_GROUND;
    case CharacterBase::EGroundState::OnSteepGround: return UJOLT_GROUND_ON_STEEP_GROUND;
    case CharacterBase::EGroundState::NotSupported: return UJOLT_GROUND_NOT_SUPPORTED;
    default: return UJOLT_GROUND_IN_AIR;
    }
}

uint32_t ujolt_character_contacts(const ujolt_character *character, ujolt_character_contact *out, uint32_t capacity) {
    uint32_t count = 0;
    for (const CharacterContact &contact : character->character->GetActiveContacts()) {
        // Sensors are not solid to the character (Jolt only notifies), and
        // one it merely overlaps is never even routed through the solver
        // that would discard it: leave them out, they are not geometry.
        if (contact.mWasDiscarded || contact.mIsSensorB) continue;
        if (count++ >= capacity) continue;
        ujolt_character_contact &c = out[count - 1];
        c.user_data = contact.mUserData;
        store3(c.position, Vec3(contact.mPosition));
        store3(c.normal, contact.mContactNormal);
        c.distance = contact.mDistance;
        c.is_dynamic = contact.mMotionTypeB == EMotionType::Dynamic ? 1 : 0;
        c.had_collision = contact.mHadCollision ? 1 : 0;
    }
    return count;
}

ujolt_body_id ujolt_character_inner_body(const ujolt_character *character) {
    return character->innerBody;
}

} // extern "C"

// MARK: - Ragdolls

namespace {

/// The constraint joining a part to its parent. Every one is a
/// SwingTwistConstraint (nothing else is ever put in mToParent); nullptr
/// for the root, which has none.
SwingTwistConstraint *jointOf(const ujolt_ragdoll *ragdoll, size_t part) {
    const int index = ragdoll->settings->GetConstraintIndexForBodyIndex(int(part));
    if (index < 0) return nullptr;
    return static_cast<SwingTwistConstraint *>(ragdoll->ragdoll->GetConstraint(index));
}

/// The parts a `part` argument names: that one, or all of them for -1.
bool partRange(const ujolt_ragdoll *ragdoll, int32_t part, size_t &first, size_t &end) {
    const size_t count = ragdoll->parts.size();
    if (part < 0) {
        first = 0;
        end = count;
        return true;
    }
    if (size_t(part) >= count) return false;
    first = size_t(part);
    end = first + 1;
    return true;
}

/// Wakes whatever rests on the parts before they leave the world (Jolt
/// would let it hover asleep), as ujolt_world_remove_body does.
void wakeAroundRagdoll(ujolt_ragdoll *ragdoll) {
    if (!ragdoll->active) return;
    AABox bounds = ragdoll->ragdoll->GetWorldSpaceBounds();
    if (!bounds.IsValid()) return;
    bounds.ExpandBy(Vec3::sReplicate(0.05f));
    ragdoll->world->system.GetBodyInterface().ActivateBodiesInAABox(bounds, BroadPhaseLayerFilter(), ObjectLayerFilter());
}


} // namespace

extern "C" {

ujolt_ragdoll *ujolt_world_add_ragdoll(ujolt_world *world, const ujolt_ragdoll_desc *desc) {
    if (desc == nullptr || desc->parts == nullptr || desc->part_count == 0) return nullptr;
    const uint32_t count = desc->part_count;
    // One root, part 0; every other part's parent before it (Jolt's
    // skeleton algorithms assume the order).
    for (uint32_t i = 0; i < count; ++i) {
        const ujolt_ragdoll_part &p = desc->parts[i];
        if (i == 0 ? p.parent != -1 : (p.parent < 0 || uint32_t(p.parent) >= i)) return nullptr;
        if (p.shape == UJOLT_SHAPE_CONVEX_HULL) return nullptr;
    }

    Ref<Skeleton> skeleton = new Skeleton();
    for (uint32_t i = 0; i < count; ++i) {
        const ujolt_ragdoll_part &p = desc->parts[i];
        skeleton->AddJoint(p.name ? p.name : "", int(p.parent));
    }
    if (!skeleton->AreJointsCorrectlyOrdered()) return nullptr;

    Ref<RagdollSettings> settings = new RagdollSettings();
    settings->mSkeleton = skeleton;
    settings->mParts.resize(count);
    std::vector<Mat44> neutral(count);
    const ObjectLayer layer = objectLayer(desc->layer, true);
    for (uint32_t i = 0; i < count; ++i) {
        const ujolt_ragdoll_part &p = desc->parts[i];
        RefConst<Shape> shape = makePrimitiveShape(p.shape, p.radius, p.half_height, p.half_extents, nullptr, 0);
        if (shape == nullptr) return nullptr;
        const Vec3 offset = v3(p.shape_offset);
        const Quat shapeRotation = q4(p.shape_rotation);
        if (offset.LengthSq() > 1e-12f || shapeRotation.GetXYZ().LengthSq() > 1e-12f) {
            shape = new RotatedTranslatedShape(offset, shapeRotation, shape);
        }
        RagdollSettings::Part &part = settings->mParts[i];
        part.SetShape(shape);
        part.mPosition = RVec3(v3(p.position));
        part.mRotation = q4(p.rotation);
        neutral[i] = Mat44::sRotationTranslation(part.mRotation, Vec3(part.mPosition));
        // Kinematic until the game says otherwise; the motion properties
        // exist either way, so a part can switch at any time.
        part.mMotionType = EMotionType::Kinematic;
        part.mAllowDynamicOrKinematic = true;
        part.mObjectLayer = layer;
        part.mUserData = desc->user_data;
        part.mGravityFactor = desc->gravity_factor;
        part.mLinearDamping = orDefault(desc->linear_damping, 0.05f);
        part.mAngularDamping = orDefault(desc->angular_damping, 0.05f);
        part.mMaxLinearVelocity = orDefault(desc->max_linear_velocity, 50.0f);
        part.mFriction = orDefault(p.friction, 0.5f);
        if (p.mass > 0.0f) {
            part.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
            part.mMassPropertiesOverride.mMass = p.mass;
        }
        if (world->continuousCollision) part.mMotionQuality = EMotionQuality::LinearCast;

        if (p.parent >= 0) {
            // Both frames at this part's pivot in the neutral pose; Jolt
            // converts them into each body's local space at creation.
            Ref<SwingTwistConstraintSettings> joint = new SwingTwistConstraintSettings();
            joint->mSpace = EConstraintSpace::WorldSpace;
            joint->mPosition1 = part.mPosition;
            joint->mPosition2 = part.mPosition;
            Vec3 twist = v3(p.twist_axis);
            twist = twist.LengthSq() > 1e-12f ? twist.Normalized() : Vec3::sAxisY();
            Vec3 plane = v3(p.plane_axis);
            plane -= twist * plane.Dot(twist);
            plane = plane.LengthSq() > 1e-12f ? plane.Normalized() : twist.GetNormalizedPerpendicular();
            // The parent's frame defaults to the part's, centring the limits
            // on the neutral pose; a rig hands in a rotated one to centre
            // them elsewhere (mid-flexion for a hinge).
            Vec3 parentTwist = v3(p.parent_twist_axis);
            Vec3 parentPlane = v3(p.parent_plane_axis);
            if (parentTwist.LengthSq() > 1e-12f) {
                parentTwist = parentTwist.Normalized();
                parentPlane -= parentTwist * parentPlane.Dot(parentTwist);
                parentPlane = parentPlane.LengthSq() > 1e-12f ? parentPlane.Normalized() : parentTwist.GetNormalizedPerpendicular();
            } else {
                parentTwist = twist;
                parentPlane = plane;
            }
            joint->mTwistAxis1 = parentTwist;
            joint->mTwistAxis2 = twist;
            joint->mPlaneAxis1 = parentPlane;
            joint->mPlaneAxis2 = plane;
            joint->mNormalHalfConeAngle = DegreesToRadians(std::clamp(p.normal_half_cone_deg, 0.0f, 180.0f));
            joint->mPlaneHalfConeAngle = DegreesToRadians(std::clamp(p.plane_half_cone_deg, 0.0f, 180.0f));
            const float twistMin = std::clamp(p.twist_min_deg, -180.0f, 180.0f);
            const float twistMax = std::clamp(p.twist_max_deg, -180.0f, 180.0f);
            joint->mTwistMinAngle = DegreesToRadians(std::min(twistMin, twistMax));
            joint->mTwistMaxAngle = DegreesToRadians(std::max(twistMin, twistMax));
            joint->mMaxFrictionTorque = std::max(p.friction_torque, 0.0f);
            const MotorSettings motor(ESpringMode::FrequencyAndDamping, orDefault(p.motor_frequency, 20.0f),
                                      orDefault(p.motor_damping, 2.0f), FLT_MAX, orDefault(p.max_torque, 500.0f));
            joint->mSwingMotorSettings = motor;
            joint->mTwistMotorSettings = motor;
            part.mToParent = joint;
        }
    }

    // Jolt's recipe: bound the parent/child mass ratios and raise parent
    // inertias so the chain does not blow up, filter parent/child (and
    // overlapping-at-rest) pairs, then solve the root's joints first.
    if (!settings->Stabilize()) return nullptr;
    settings->DisableParentChildCollisions(neutral.data(), 0.0f);
    if (desc->disabled_pairs != nullptr && desc->disabled_pair_count > 0) {
        // DisableParentChildCollisions gave every part the same table.
        // The settings own the table they just built; it is not shared yet.
        auto *table = const_cast<GroupFilterTable *>(static_cast<const GroupFilterTable *>(settings->mParts[0].mCollisionGroup.GetGroupFilter()));
        if (table != nullptr) {
            for (uint32_t k = 0; k < desc->disabled_pair_count; ++k) {
                const int32_t a = desc->disabled_pairs[2 * k], b = desc->disabled_pairs[2 * k + 1];
                if (a < 0 || b < 0 || uint32_t(a) >= count || uint32_t(b) >= count || a == b) continue;
                table->DisableCollision(CollisionGroup::SubGroupID(a), CollisionGroup::SubGroupID(b));
            }
        }
    }
    settings->CalculateBodyIndexToConstraintIndex();
    settings->CalculateConstraintPriorities();

    Ragdoll *created = settings->CreateRagdoll(world->nextRagdollGroup++, desc->user_data, &world->system);
    if (created == nullptr) return nullptr;
    ujolt_ragdoll *ragdoll = new ujolt_ragdoll();
    ragdoll->world = world;
    ragdoll->skeleton = skeleton;
    ragdoll->settings = settings;
    ragdoll->ragdoll = created;
    ragdoll->parts.reserve(count);
    ragdoll->motors.assign(count, UJOLT_MOTOR_OFF);
    ragdoll->kinematicPose = neutral;
    {
        std::lock_guard<std::mutex> guard(world->recordMutex);
        for (uint32_t i = 0; i < count; ++i) {
            const ujolt_body_id id = created->GetBodyID(int(i)).GetIndexAndSequenceNumber();
            ragdoll->parts.push_back(id);
            // Kinematic like the body: the contact convention (the dynamic
            // body is A) reads the record's motion type, which follows
            // set_part_dynamic.
            BodyRecord record{desc->user_data, false, EMotionType::Kinematic};
            record.ragdoll = true;
            world->records[id] = record;
        }
    }
    if (desc->start_active != 0) {
        created->AddToPhysicsSystem(EActivation::Activate);
        ragdoll->active = true;
    }
    world->broadPhaseDirty = true;
    world->ragdolls.insert(ragdoll);
    return ragdoll;
}

void ujolt_world_remove_ragdoll(ujolt_world *world, ujolt_ragdoll *ragdoll) {
    if (ragdoll == nullptr || world->ragdolls.erase(ragdoll) == 0) return;
    wakeAroundRagdoll(ragdoll);
    world->releaseRagdoll(ragdoll, true);
}

uint32_t ujolt_ragdoll_part_count(const ujolt_ragdoll *ragdoll) {
    return uint32_t(ragdoll->parts.size());
}

void ujolt_ragdoll_set_active(ujolt_ragdoll *ragdoll, int32_t active) {
    if ((active != 0) == ragdoll->active) return;
    if (active != 0) {
        ragdoll->ragdoll->AddToPhysicsSystem(EActivation::Activate);
        ragdoll->active = true;
    } else {
        wakeAroundRagdoll(ragdoll);
        // Constraints go before bodies; Jolt orders that itself.
        ragdoll->ragdoll->RemoveFromPhysicsSystem();
        ragdoll->active = false;
    }
    ragdoll->world->broadPhaseDirty = true;
}

int32_t ujolt_ragdoll_is_active(const ujolt_ragdoll *ragdoll) {
    return ragdoll->active ? 1 : 0;
}

void ujolt_ragdoll_set_pose(ujolt_ragdoll *ragdoll, const float *world_matrices, int32_t reset_velocities) {
    if (world_matrices == nullptr) return;
    BodyInterface &bi = ragdoll->world->system.GetBodyInterface();
    for (size_t i = 0; i < ragdoll->parts.size(); ++i) {
        const BodyID id(ragdoll->parts[i]);
        const Mat44 m = mat44(world_matrices + i * 16);
        // Never through Activate here: a part of an inactive ragdoll is not
        // in the broad phase, which activation asserts.
        bi.SetPositionAndRotation(id, RVec3(m.GetTranslation()), rotationOf(m), EActivation::DontActivate);
        if (reset_velocities != 0) bi.SetLinearAndAngularVelocity(id, Vec3::sZero(), Vec3::sZero());
    }
    // The constraints' remembered impulses belong to the old pose.
    ragdoll->ragdoll->ResetWarmStart();
    if (ragdoll->active) ragdoll->ragdoll->Activate();
}

void ujolt_ragdoll_set_velocities(ujolt_ragdoll *ragdoll, const float *linear, const float *angular) {
    if (linear == nullptr) return;
    // Through the lock rather than the BodyInterface, which would wake a
    // body given a velocity (and assert on one of an inactive ragdoll).
    const BodyLockInterface &lockInterface = ragdoll->world->system.GetBodyLockInterface();
    for (size_t i = 0; i < ragdoll->parts.size(); ++i) {
        BodyLockWrite lock(lockInterface, BodyID(ragdoll->parts[i]));
        if (!lock.Succeeded()) continue;
        Body &body = lock.GetBody();
        body.SetLinearVelocityClamped(v3(linear + i * 3));
        body.SetAngularVelocityClamped(angular ? v3(angular + i * 3) : Vec3::sZero());
    }
    // A sleeping body's velocity is dropped when it wakes on its own.
    if (ragdoll->active) ragdoll->ragdoll->Activate();
}

void ujolt_ragdoll_set_kinematic_pose(ujolt_ragdoll *ragdoll, const float *world_matrices) {
    if (world_matrices == nullptr) return;
    for (size_t i = 0; i < ragdoll->parts.size(); ++i) {
        ragdoll->kinematicPose[i] = mat44(world_matrices + i * 16);
    }
    ragdoll->kinematicPending = true;
}

void ujolt_ragdoll_drive_motors(ujolt_ragdoll *ragdoll, const float *world_matrices) {
    if (world_matrices == nullptr) return;
    // Jolt's own DriveToPoseUsingMotors would switch every motor on; only
    // the joints the game put in position mode take a target here.
    const Skeleton &skeleton = *ragdoll->skeleton;
    for (size_t i = 0; i < ragdoll->parts.size(); ++i) {
        if (ragdoll->motors[i] != UJOLT_MOTOR_POSITION) continue;
        SwingTwistConstraint *joint = jointOf(ragdoll, i);
        if (joint == nullptr) continue;
        const int parent = skeleton.GetJoint(int(i)).mParentJointIndex;
        // The target is body 2 (this part) relative to body 1 (its parent):
        // this part's world rotation taken into the parent's frame.
        const Quat own = rotationOf(world_matrices + i * 16);
        const Quat parentRotation = rotationOf(world_matrices + size_t(parent) * 16);
        joint->SetTargetOrientationBS(parentRotation.Conjugated() * own);
    }
}

void ujolt_ragdoll_set_part_dynamic(ujolt_ragdoll *ragdoll, int32_t part, int32_t dynamic) {
    size_t first, end;
    if (!partRange(ragdoll, part, first, end)) return;
    BodyInterface &bi = ragdoll->world->system.GetBodyInterface();
    const EMotionType motion = dynamic != 0 ? EMotionType::Dynamic : EMotionType::Kinematic;
    const EActivation activation = ragdoll->active ? EActivation::Activate : EActivation::DontActivate;
    std::lock_guard<std::mutex> guard(ragdoll->world->recordMutex);
    for (size_t i = first; i < end; ++i) {
        const BodyID id(ragdoll->parts[i]);
        if (bi.GetMotionType(id) == motion) continue;
        bi.SetMotionType(id, motion, activation);
        // A kinematic body keeps integrating whatever velocity it has: a
        // part handed back to the animation must not fly on by itself.
        if (motion == EMotionType::Kinematic) bi.SetLinearAndAngularVelocity(id, Vec3::sZero(), Vec3::sZero());
        auto record = ragdoll->world->records.find(ragdoll->parts[i]);
        if (record != ragdoll->world->records.end()) record->second.motion = motion;
    }
}

void ujolt_ragdoll_set_allow_sleeping(ujolt_ragdoll *ragdoll, int32_t allow) {
    const BodyLockInterface &lockInterface = ragdoll->world->system.GetBodyLockInterface();
    for (ujolt_body_id id : ragdoll->parts) {
        BodyLockWrite lock(lockInterface, BodyID(id));
        if (lock.Succeeded()) lock.GetBody().SetAllowSleeping(allow != 0);
    }
    // A part already asleep stays asleep until woken.
    if (allow == 0 && ragdoll->active) ragdoll->ragdoll->Activate();
}

int32_t ujolt_ragdoll_part_is_awake(const ujolt_ragdoll *ragdoll, int32_t part) {
    if (part < 0 || size_t(part) >= ragdoll->parts.size() || !ragdoll->active) return 0;
    return ragdoll->world->system.GetBodyInterface().IsActive(BodyID(ragdoll->parts[size_t(part)])) ? 1 : 0;
}

int32_t ujolt_ragdoll_part_is_dynamic(const ujolt_ragdoll *ragdoll, int32_t part) {
    if (part < 0 || size_t(part) >= ragdoll->parts.size()) return 0;
    const BodyID id(ragdoll->parts[size_t(part)]);
    return ragdoll->world->system.GetBodyInterface().GetMotionType(id) == EMotionType::Dynamic ? 1 : 0;
}

void ujolt_ragdoll_set_motors(ujolt_ragdoll *ragdoll, int32_t part, ujolt_motor_mode mode, float frequency, float damping,
                              float max_torque, float friction_torque) {
    size_t first, end;
    if (!partRange(ragdoll, part, first, end)) return;
    const bool position = mode == UJOLT_MOTOR_POSITION;
    const EMotorState state = position ? EMotorState::Position : EMotorState::Off;
    for (size_t i = first; i < end; ++i) {
        SwingTwistConstraint *joint = jointOf(ragdoll, i);
        if (joint == nullptr) continue;
        ragdoll->motors[i] = position ? UJOLT_MOTOR_POSITION : UJOLT_MOTOR_OFF;
        joint->SetSwingMotorState(state);
        joint->SetTwistMotorState(state);
        for (MotorSettings *motor : {&joint->GetSwingMotorSettings(), &joint->GetTwistMotorSettings()}) {
            if (frequency > 0.0f) motor->mSpringSettings.mFrequency = frequency;
            if (damping > 0.0f) motor->mSpringSettings.mDamping = damping;
            if (max_torque > 0.0f) motor->SetTorqueLimit(max_torque);
        }
        if (friction_torque > 0.0f) joint->SetMaxFrictionTorque(friction_torque);
    }
}

uint32_t ujolt_ragdoll_read_pose(const ujolt_ragdoll *ragdoll, float *world_matrices, uint32_t capacity) {
    if (world_matrices == nullptr) return 0;
    const BodyInterface &bi = ragdoll->world->system.GetBodyInterface();
    const uint32_t count = std::min<uint32_t>(capacity, uint32_t(ragdoll->parts.size()));
    for (uint32_t i = 0; i < count; ++i) {
        RVec3 position;
        Quat rotation;
        bi.GetPositionAndRotation(BodyID(ragdoll->parts[i]), position, rotation);
        storeMat44(world_matrices + i * 16, Mat44::sRotationTranslation(rotation, Vec3(position)));
    }
    return count;
}

int32_t ujolt_ragdoll_read_joint_rotation(const ujolt_ragdoll *ragdoll, int32_t part, float out_quat[4]) {
    SwingTwistConstraint *joint = jointOf(ragdoll, part);
    if (joint == nullptr) return 0;
    store4(out_quat, joint->GetRotationInConstraintSpace());
    return 1;
}

void ujolt_ragdoll_add_impulse(ujolt_ragdoll *ragdoll, int32_t part, const float impulse[3], const float world_point[3]) {
    // Jolt applies it to dynamic bodies only and wakes them, which a part
    // of an inactive ragdoll (not in the broad phase) cannot be.
    if (!ragdoll->active || part < 0 || size_t(part) >= ragdoll->parts.size()) return;
    BodyInterface &bi = ragdoll->world->system.GetBodyInterface();
    const BodyID id(ragdoll->parts[size_t(part)]);
    if (world_point == nullptr) bi.AddImpulse(id, v3(impulse));
    else bi.AddImpulse(id, v3(impulse), RVec3(v3(world_point)));
}

void ujolt_ragdoll_set_gravity_factor(ujolt_ragdoll *ragdoll, int32_t part, float factor) {
    size_t first, end;
    if (!partRange(ragdoll, part, first, end)) return;
    BodyInterface &bi = ragdoll->world->system.GetBodyInterface();
    for (size_t i = first; i < end; ++i) bi.SetGravityFactor(BodyID(ragdoll->parts[i]), factor);
}

} // extern "C"
