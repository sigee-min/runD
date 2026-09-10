#pragma once

#include "../local.hpp"

#include "../../../allocation.hpp"
#include "src/accel/backend/ops/table.hpp"
#include "src/accel/backend/token.hpp"
#include "src/accel/kernel/prepared/interface/api.hpp"
#include "src/accel/kernel/prepared/model.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <thread>

namespace rund_node_test_pipeline_residency::native_sliding {

namespace accel = rund::node::accel::detail;
namespace prepared = rund::node::accel::detail::prepared;

struct FakeSlidingBackend final {
  std::mutex gate{};
  std::condition_variable ready{};
  accel::KernelCompletion completion{};
  void *user{};
  std::array<std::uint32_t, accel::ResidencyWindowLocalCapacity> locals{};
  std::size_t local_count{};
  std::uint32_t seeded{};
  std::uint32_t delay_generation{};
  const void *descriptor_owner{};
  std::uint64_t descriptor_plan{};
  std::uint64_t descriptor_token{};
  std::uint64_t descriptor_run{};
  std::uint64_t descriptor_generation{};
  std::atomic_uint64_t submission_count{};
  std::int32_t generation_delta{};
  bool submitted{};
  bool released{};
  bool delay_open{};
  bool inline_completion{};
  bool reject_after_inline{};
  bool cross_thread_early_completion{};
  bool reject_after_cross_thread{};
  bool early_completion_returned{};
  bool unknown_terminal{};
  bool known_failure{};
  const char *failure_reason{"compute_backend_failed"};
  bool block_seed{};
  bool seed_entered{};
  bool seed_open{};
  bool block_submit_return{};
  bool submit_entered{};
  bool submit_open{};
  std::shared_ptr<FakeSlidingBackend> cross_completion_backend{};
  void (*nested_submit)(void *) noexcept {};
  void *nested_user{};
  bool stop{};
  std::thread worker{};

  ~FakeSlidingBackend();
};

[[nodiscard]] accel::KernelResult FakeResult(
    std::uint32_t generation, std::size_t local_count, bool unknown = false,
    bool known_failure = false,
    const char *failure_reason = "compute_backend_failed") noexcept;

[[nodiscard]] accel::BackendResidencySlidingCapability FakeCapability(
    const std::shared_ptr<void> &raw,
    accel::ResidencySlidingMemory memory) noexcept;

[[nodiscard]] accel::BackendResidencySlidingCapability RejectedCapability(
    const std::shared_ptr<void> &raw,
    accel::ResidencySlidingMemory memory) noexcept;

[[nodiscard]] rund::AccelCheck FakeSeed(const std::shared_ptr<void> &raw,
                                         std::uint32_t generation) noexcept;

[[nodiscard]] rund::AccelCheck FakeSubmit(
    const std::shared_ptr<void> &raw,
    const accel::BackendResidencySlidingDescriptor &descriptor,
    accel::KernelCompletion completion, void *user, accel::KernelTiming timing,
    accel::PipelineSubmitMode mode,
    std::span<const std::uint32_t> locals) noexcept;

void BackendWorker(const std::shared_ptr<FakeSlidingBackend> &backend);

struct SlidingWait {
  std::mutex gate{};
  std::condition_variable ready{};
  std::array<std::uint64_t, 4u> slot_turn{};
  std::array<std::uint64_t, 4u> slot_last{
      std::numeric_limits<std::uint64_t>::max(),
      std::numeric_limits<std::uint64_t>::max(),
      std::numeric_limits<std::uint64_t>::max(),
      std::numeric_limits<std::uint64_t>::max()};
  std::uint64_t coordinate_count{};
  std::uint64_t releases{};
  std::size_t stride{4u};
  accel::PreparedResidencySlidingControl *control{};
  std::uint64_t fail_coordinate{std::numeric_limits<std::uint64_t>::max()};
  bool reentrant_wake{};
  bool reentrant_woke{};
  std::size_t reentrant_wake_limit{1u};
  std::size_t reentrant_wake_count{};
  bool pending_once{};
  bool pending_returned{};
  std::size_t pending_remaining{};
  std::uint64_t block_coordinate{std::numeric_limits<std::uint64_t>::max()};
  bool project_entered{};
  bool project_open{};
  std::uint64_t block_release_coordinate{
      std::numeric_limits<std::uint64_t>::max()};
  bool release_entered{};
  bool release_open{};
  std::uint64_t block_returned_coordinate{
      std::numeric_limits<std::uint64_t>::max()};
  bool returned_entered{};
  bool returned_open{};
  bool returned_opens_on_peer{};
  std::uint64_t returned_count{};
  bool delay_opened{};
  bool advanced_past_delay{};
  bool valid{true};
  bool final{};
  std::atomic_bool final_observed{};
  std::atomic_uint32_t release_started_mask{};
  std::atomic_uint64_t release_coordinate_mask{};
  accel::BackendResidencySlidingFinal evidence{};
};

[[nodiscard]] accel::PreparedResidencySlidingProjection ProjectSliding(
    void *raw, std::uint64_t coordinate, std::uint64_t turn, std::uint8_t slot,
    accel::PreparedResidencySlidingSelection &selection) noexcept;

void CompleteSlidingRelease(
    void *raw, accel::PreparedResidencySlidingRelease &&release) noexcept;

[[nodiscard]] bool CompleteSlidingReturned(void *raw, std::uint64_t coordinate,
                                            std::uint64_t turn,
                                            std::uint8_t slot) noexcept;

void CompleteSlidingFinal(
    void *raw, accel::BackendResidencySlidingFinal &&final) noexcept;

struct SlidingFixture final {
  accel::BackendOps ops{};
  std::array<std::shared_ptr<FakeSlidingBackend>, 4u> backends{};
  std::array<std::shared_ptr<prepared::PipelineState>, 4u> states{};
  std::array<accel::PreparedResidencySlidingRole, 4u> roles{};
  std::shared_ptr<std::uint8_t> context_owner{};
  std::shared_ptr<accel::PickToken> pick{};
  rund::AccelContext context{};
  accel::PreparedResidencySlidingControl control{};
  std::size_t role_count{};

  SlidingFixture();

  [[nodiscard]] bool BuildRoles(std::size_t count = 4u);
  [[nodiscard]] bool Prepare(std::size_t count = 4u);
  void StartWorkers();
};

[[nodiscard]] accel::PreparedResidencySlidingRequest MakeRequest(
    const SlidingFixture &fixture, SlidingWait &wait, std::uint64_t q,
    std::uint64_t plan = 101u, std::uint64_t token = 202u,
    std::uint64_t generation = 303u) noexcept;

[[nodiscard]] bool WaitForFinal(SlidingWait &wait);
[[nodiscard]] bool WaitForReleaseMask(SlidingWait &wait, std::uint32_t mask);

struct ReentrantWait final : SlidingWait {
  SlidingFixture *fixture{};
  accel::BackendResidencySlidingFinal first{};
  std::size_t finals{};
  bool resubmit_ok{};
  bool resubmit_returned{};
};

void CompleteReentrantFinal(
    void *raw, accel::BackendResidencySlidingFinal &&final) noexcept;

struct NestedSubmit final {
  SlidingFixture *fixture{};
  SlidingWait *wait{};
  bool returned{};
  bool ok{};
};

void SubmitNested(void *raw) noexcept;

enum class SuppressionStage : std::uint8_t { Project, Seed };

[[nodiscard]] bool AdmissionRejectNoAllocationCase();
[[nodiscard]] bool SlidingCase(std::uint64_t q);
[[nodiscard]] bool StrideCase(std::size_t roles);
[[nodiscard]] bool InvalidMaskCase();

[[nodiscard]] bool ProjectionFailureCase(bool accepted_sibling);
[[nodiscard]] bool InlineContradictionCase();
[[nodiscard]] bool CrossThreadEarlyContradictionCase();
[[nodiscard]] bool ActiveSelfRetainCase();

[[nodiscard]] bool SuppressionCase(SuppressionStage stage);
[[nodiscard]] bool DelayedSubmitReturnCase();
[[nodiscard]] bool ReleaseHandoffCase();
[[nodiscard]] bool ReturnedPressureCase();
[[nodiscard]] bool ReturnedProgressRearmCase();
[[nodiscard]] bool CrossSlotInlineCase();
[[nodiscard]] bool WrongControlGenerationCase();

[[nodiscard]] bool ReentrantFinalCase();
[[nodiscard]] bool NestedInlineCase();
[[nodiscard]] bool ReverseFailureCase();
[[nodiscard]] bool MultiReentrantWakeCase();

} // namespace rund_node_test_pipeline_residency::native_sliding
