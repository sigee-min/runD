#pragma once

#include "../local.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>

namespace rund::node::accel::detail::metal_persistent_sliding {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

inline constexpr std::uint64_t OwnerMagic = 0x4d'50'53'4c'49'44'45'31ull;
inline constexpr std::uint64_t DoneWaitMilliseconds = 30'000u;
inline constexpr std::size_t SlotCount = 2u;

// Validation keeps one issue taxonomy and one transient normalized role
// record.  They are declaration-seam values only: Owner remains the sole
// retained lifecycle model, while the validation leaves below own each
// dimension's implementation.
enum class RequestIssue : std::uint32_t {
  Valid = 0u,
  PlanIdentity = 1u,
  Token = 2u,
  Generation = 3u,
  OwnerNonce = 4u,
  CoordinateCount = 5u,
  TailLocalCount = 6u,
  Admission = 7u,
  Final = 8u,
  User = 9u,
  Memory = 10u,
  Mode = 11u,
  WidthZero = 12u,
  WidthCapacity = 13u,
  RoleSlot = 0x100u,
  RoleLocalZero = 0x101u,
  RoleLocalCapacity = 0x102u,
  RoleControlGeneration = 0x103u,
  RoleControlStride = 0x104u,
  RoleDescriptorGeneration = 0x105u,
  RoleDescriptorStride = 0x106u,
  RoleSequence = 0x107u,
  RoleAdapter = 0x108u,
  RoleDirectAggregate = 0x109u,
  RoleStateCount = 0x10au,
  RoleStates = 0x10bu,
  RoleRecurrence = 0x10cu,
  RoleSpatialRoute = 0x10du,
  RoleSpatialProof = 0x10eu,
  RoleSpatialStateCount = 0x10fu,
  RoleSpatialStates = 0x110u,
  RoleSpatialRecurrence = 0x111u,
  RoleSelectable = 0x112u,
  RoleGuard = 0x113u,
  RoleGuardContents = 0x114u,
  RoleControl = 0x115u,
  RoleControlContents = 0x116u,
  RoleSubmissionReady = 0x117u,
  RoleSlidingReady = 0x118u,
  RoleDescriptor = 0x119u,
  RoleCommand = 0x11au,
  RolePipeline = 0x11bu,
  RoleAdapterMismatch = 0x11cu,
  LocalStepBounds = 0x200u,
  LocalSpatialBounds = 0x201u,
  LocalSpatialMatch = 0x202u,
  LocalDuplicate = 0x203u,
  TailAdapter = 0x300u,
  TailQueue = 0x301u,
  AdapterQuarantined = 0x302u,
  TailCapacity = 0x303u,
  GenerationRange = 0x304u,
};

struct NativeRole final {
  const MetalSequence *sequence{};
  std::array<std::uint32_t, ResidencyWindowLocalCapacity> locals{};
  std::size_t local_count{};
  std::uint32_t first_control_generation{};
  std::uint32_t control_generation_stride{};
  std::uint64_t first_descriptor_generation{};
  std::uint64_t descriptor_generation_stride{};
  std::uint8_t slot{};
  bool pipeline_ok{};
};

[[nodiscard]] std::uint64_t
request_issue_key(RequestIssue issue, std::size_t slot = std::size_t(-1),
                  std::size_t local = std::size_t(-1)) noexcept;
[[nodiscard]] std::uint64_t valid_request_issue() noexcept;
[[nodiscard]] NativeRole
native_role(const PreparedResidencyPersistentSlidingRole &) noexcept;
[[nodiscard]] std::uint64_t
first_invalid_structure(std::span<const NativeRole>, std::size_t width,
                        std::uint64_t coordinate_count, ResidencySlidingMemory,
                        PersistentResidencySlidingMode,
                        MetalAdapter *&) noexcept;

struct Coordinate final {
  PersistentResidencySlidingServiceIdentity identity{};
  rund::AccelCheck admission{true, "ok"};
  MetalSequence *sequence{};
  std::array<std::uint32_t, ResidencyWindowLocalCapacity> locals{};
  std::size_t local_count{};
  std::uint64_t ready_value{};
  std::uint64_t done_value{};
};

struct Owner final : std::enable_shared_from_this<Owner> {
  std::mutex gate{};
  std::uint64_t magic{OwnerMagic};
  std::uint64_t owner_nonce{};
  std::uint64_t event_base{};
  MetalAdapter *adapter{};
  PersistentResidencySlidingRequest prepared{};
  PersistentResidencySlidingRequest pending_request{};
  PersistentResidencySlidingRequest rollback_request{};
  bool native_pending{};
  id<MTLCommandBuffer> rollback_command = nil;
  bool native_prepared{};
  PersistentResidencySlidingCapability capability{};
  id<MTLCommandBuffer> command = nil;
  std::array<id<MTLSharedEvent>, PersistentResidencySlidingCapacity> ready{};
  id<MTLSharedEvent> done = nil;
  // Coordinates are only live until their chunk is waited and acknowledged.
  // Reusing these two slots keeps backend-owned storage independent of Q.
  std::array<Coordinate, SlotCount> coordinates{};
  std::shared_ptr<void> quarantine{};
  std::uint64_t encoded_intermediate_bytes{};
  std::uint64_t submit_begin_ns{};
  std::uint64_t run_event_base{};
  bool native_retired{};
  bool quarantine_ticket{};
  MetalPersistentResidencySlidingDiagnosticTrace first_failure_trace{};
  MetalPersistentResidencySlidingDiagnosticTrace last_wait_trace{};
  bool first_failure_trace_recorded{};
  struct Sealed final {
    MetalPersistentResidencySlidingDiagnostics value{};
    bool valid{};
  } sealed{};
};

struct Binding final {
  std::shared_ptr<void> native{};
  std::shared_ptr<Owner> owner{};
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t owner_nonce{};
};

void record_first_failure(Owner &,
                          MetalPersistentResidencySlidingDiagnosticStage,
                          std::uint64_t,
                          MetalPersistentResidencySlidingDiagnosticPredicate,
                          std::uint64_t, std::uint64_t) noexcept;
void record_first_failure(
    Owner &, const MetalPersistentResidencySlidingDiagnosticTrace &) noexcept;
void record_wait_trace(
    Owner &, const MetalPersistentResidencySlidingDiagnosticTrace &) noexcept;

[[nodiscard]] std::shared_ptr<Owner>
owner_of(const std::shared_ptr<void> &) noexcept;
[[nodiscard]] Binding bind(PersistentResidencySlidingControl &) noexcept;
[[nodiscard]] bool valid_binding(const Binding &, const Owner &,
                                 PersistentResidencySlidingControl &) noexcept;
[[nodiscard]] bool
same_prepared_request(const Owner &,
                      const PersistentResidencySlidingRequest &) noexcept;
[[nodiscard]] std::uint64_t
first_invalid_request(const PersistentResidencySlidingRequest &,
                      MetalAdapter *&) noexcept;
[[nodiscard]] bool
stage_rearm(Owner &, const PersistentResidencySlidingRequest &) noexcept;
[[nodiscard]] bool prepare_rearm(Owner &) noexcept;
[[nodiscard]] bool commit_rearm(Owner &) noexcept;
void abort_rearm(Owner &) noexcept;
[[nodiscard]] rund::AccelCheck
preflight(const PersistentResidencySlidingRequest &, MetalAdapter *&) noexcept;
[[nodiscard]] rund::AccelCheck encode(Owner &) noexcept;
[[nodiscard]] rund::AccelCheck encode_chunk(Owner &, std::uint64_t first,
                                            std::uint64_t count) noexcept;
[[nodiscard]] bool coordinate_at(const Owner &, std::uint64_t,
                                 Coordinate &) noexcept;
[[nodiscard]] rund::AccelCheck
encode_selection(id<MTLComputeCommandEncoder>, MetalSequence &,
                 std::span<const std::uint32_t>) noexcept;

[[nodiscard]] PersistentResidencySlidingSubmitResult
submit_result(const PersistentResidencySlidingRequest &,
              PersistentResidencySlidingControl &) noexcept;
[[nodiscard]] rund::AccelCheck
signal_ready(PersistentResidencySlidingControl &,
             const PersistentResidencySlidingReadySignal &) noexcept;
[[nodiscard]] rund::AccelCheck
wait_done(PersistentResidencySlidingControl &,
          const PersistentResidencySlidingDoneWait &,
          PersistentResidencySlidingDoneObservation &) noexcept;
[[nodiscard]] rund::AccelCheck
acknowledge_done(PersistentResidencySlidingControl &,
                 const PersistentResidencySlidingAcknowledgeDone &) noexcept;
[[nodiscard]] rund::AccelCheck
fail_service(PersistentResidencySlidingControl &,
             const PersistentResidencySlidingServiceFailure &) noexcept;

void release_known_claims(Owner &) noexcept;
void quarantine_unknown(PersistentResidencySlidingControl &) noexcept;
void quarantine_unknown(Owner &, PersistentResidencySlidingControl &) noexcept;
void quarantine_owner(Owner &) noexcept;
[[nodiscard]] PersistentResidencySlidingFinal
make_final(Owner &, const PersistentResidencySlidingControl &, rund::AccelCheck,
           NativeTerminal, std::uint64_t first_failure) noexcept;
void deliver_final(Owner &, PersistentResidencySlidingFinal &&) noexcept;

#endif

} // namespace rund::node::accel::detail::metal_persistent_sliding
