#pragma once

#include "../../../../accel/kernel/prepared/interface/residency.hpp"

#include "../../../device/residency/registry.hpp"
#include "../../../../accel/kernel/residency/persistent_sliding/request.hpp"
#include <rund/storage.hpp>

namespace rund::compute::detail::sliding_product_detail {

struct SlidingProductRun;

enum class OutputServicePhase : std::uint8_t {
  NeedDrain,
  NeedPersist,
  Done,
};

enum class OutputServiceResult : std::uint8_t {
  Ready,
  Pending,
  Failed,
};

enum class InputServiceResult : std::uint8_t {
  Ready,
  Pending,
  Failed,
};

enum class CloseRequirement : std::uint8_t {
  None,
  Required,
};

using ProjectionResult =
    node::accel::detail::PreparedResidencySlidingProjection;

// Static identity for the retained product.  Attempt credentials, prepared
// handles, snapshots, visible bytes, and terminal fields deliberately do not
// participate in this key.
struct SlidingProductKey final {
  std::array<const void *, 4u> owners{};
  std::array<std::uint64_t, 128u> scalar{};
  std::size_t scalar_count{};
  std::array<residency::FrameRegion, residency::execution::BankCapacity>
      host_input{};
  std::array<residency::FrameRegion, residency::execution::BankCapacity>
      device_input{};
  std::array<residency::FrameRegion, residency::execution::BankCapacity>
      device_output{};
  std::array<residency::FrameRegion, residency::execution::BankCapacity>
      host_output{};

  node::accel::detail::ResidencySlidingMemory memory{
      node::accel::detail::ResidencySlidingMemory::HostCoherent};

  [[nodiscard]] constexpr bool
  operator==(const SlidingProductKey &) const noexcept = default;
};

struct SlidingProductAttempt final {
  PipelineExecutionSnapshot snapshot{};
  std::array<node::accel::detail::PreparedResidencySlidingRole,
             node::accel::detail::ResidencySlidingCapacity>
      roles{};
  std::array<node::accel::detail::PreparedResidencyPersistentSlidingRole,
             node::accel::detail::PersistentResidencySlidingCapacity>
      persistent_roles{};
  std::size_t role_count{};
  std::uint32_t generation_step{};
};

struct SlidingProductOwner final {
  // Cold capacity is reserved before any persistent owner/backend allocation.
  // Once the lowering reports its exact logical payload, memory owns the
  // committed common+backend charge and capacity has been refunded.
  storage::Reservation capacity{};
  std::shared_ptr<storage::Reservation> memory{};
  residency::execution::Plan plan{};
  std::shared_ptr<const residency::execution::Plan> plan_owner{};
  std::array<std::shared_ptr<PipelineState>, residency::execution::BankCapacity>
      pipelines{};
  std::array<node::accel::detail::PreparedResidencySlidingRole,
             node::accel::detail::ResidencySlidingCapacity>
      roles{};
  std::array<node::accel::detail::PreparedResidencyPersistentSlidingRole,
             node::accel::detail::PersistentResidencySlidingCapacity>
      persistent_roles{};
  PipelineExecutionSnapshot snapshot{};
  node::accel::detail::PreparedResidencySlidingControl native{};
  std::shared_ptr<SlidingProductRun> run{};
  SlidingProductKey key{};
  bool key_sealed{};
  SlidingProductAttempt pending{};
  bool pending_valid{};
  std::size_t role_count{};
  std::uint32_t generation_step{};
  std::uint64_t prepared_lease_token{};
  std::uint32_t prepared_lease_count{};
  // Admission owns this immutable lowering choice; the live Control copies
  // it for the duration of an attempt.
  node::accel::detail::PersistentResidencySlidingMode mode{
      node::accel::detail::PersistentResidencySlidingMode::OneSubmit};
};

[[nodiscard]] Status
stage_preparation_roles(SlidingProductOwner &,
                        const PipelineExecutionSnapshot &) noexcept;
void commit_staged_roles(SlidingProductOwner &) noexcept;
void discard_staged_roles(SlidingProductOwner &) noexcept;

struct SlidingProductWork final {
  residency::execution::SlidingProjection projection{};
  std::array<residency::PageUse,
             residency::execution::WindowFootprintSourceCapacity +
                 residency::execution::UseCapacity>
      uses{};
  residency::execution::SlidingNative native{};
  std::array<OutputServicePhase, residency::execution::UseCapacity>
      output_phase{};
  Status native_status{Status::fail(Reason::PipelineInvalid)};
  residency::execution::TerminalKind native_terminal{
      residency::execution::TerminalKind::Known};
  std::size_t next_fetch{};
  std::size_t output_count{};
  bool projected{};
  bool promoted{};
  bool terminaled{};
  bool native_released{};
  bool native_may_write{};
};

enum class ServiceFaultStage : std::uint8_t {
  None,
  Init,
  Admit,
  Select,
  Ops,
  Signal,
  Wait,
};

struct ServiceFault final {
  ServiceFaultStage stage{ServiceFaultStage::None};
  std::uint32_t code{};
  std::uint64_t coordinate{};
  std::uint32_t reason{};
  std::uint64_t key{};
};

struct SlidingProductRun final
    : std::enable_shared_from_this<SlidingProductRun> {
  std::mutex gate{};
  std::condition_variable ready{};
  std::shared_ptr<void> cold_owner{};
  std::shared_ptr<SlidingProductRun> quarantine{};
  SlidingProductOwner *cold{};
  VirtualPipelineState *state{};
  VirtualBacking *input{};
  VirtualBacking *output{};
  const VirtualRunProjection *run{};
  Stats *stats{};
  residency::Pool *pool{};
  residency::execution::Sliding sliding{};
  node::accel::detail::PreparedResidencyPersistentSlidingPreparation
      persistent_preparation{};
  node::accel::detail::PersistentResidencySlidingControl persistent_control{};
  node::accel::detail::PersistentResidencySlidingFinal persistent_final{};
  std::array<SlidingProductWork, node::accel::detail::ResidencySlidingCapacity>
      work{};
  ::rund::node::hash_detail::Fnv hash{};
  VirtualExecutionResult result{};
  Status failure{Status::success()};
  std::uint64_t failure_coordinate{std::numeric_limits<std::uint64_t>::max()};
  std::uint64_t backing_read_bytes{};
  std::uint64_t backing_write_bytes{};
  std::uint64_t backing_io_ns{};
  std::uint64_t epoch_count{};
  bool recovery{};
  bool persistent_final_received{};
  bool completed{};
  std::atomic_bool poison{};
  std::array<bool, residency::execution::BankCapacity> pipeline_started{};
  std::array<bool, residency::execution::BankCapacity> pipeline_submitted{};
  ServiceFault service_fault{};
};

struct GenerationClose final {
  Status status{Status::fail(Reason::CompletionInvalid)};
  residency::execution::SlidingEvidence evidence{};
  bool closed{};
};

// Success-only publication proof.  prepare_persistent_publication() retains
// both Pipeline gates after it has frozen/rebased the generation and consumed
// the fused Authority/Sliding receipt. The terminal-pair commit then locks
// both publication owners as one externally indivisible transition.
struct PersistentPublication final {
  PersistentPublication() = default;
  PersistentPublication(const PersistentPublication &) = delete;
  PersistentPublication &operator=(const PersistentPublication &) = delete;
  PersistentPublication(PersistentPublication &&) noexcept = default;
  PersistentPublication &operator=(PersistentPublication &&) noexcept = default;

  residency::execution::SlidingEvidence evidence{};
  std::unique_lock<std::mutex> first_pipeline{};
  std::unique_lock<std::mutex> second_pipeline{};
  SlidingProductRun *state{};
  const node::accel::detail::PersistentResidencySlidingFinal *native{};
  CloseRequirement close{CloseRequirement::None};
  residency::FinalAbort abort{residency::FinalAbort::Invalid};
};

struct FetchIo final {
  Status status{Status::fail(Reason::PipelineInvalid)};
  std::uint64_t bytes{};
  std::uint64_t started_ns{};
};

struct PersistIo final {
  Status status{Status::fail(Reason::PipelineInvalid)};
  const std::byte *frame{};
  std::size_t hash_offset{};
  std::uint64_t bytes{};
  std::uint64_t started_ns{};
};

} // namespace rund::compute::detail::sliding_product_detail
