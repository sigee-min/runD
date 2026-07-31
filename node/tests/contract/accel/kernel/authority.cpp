#include "src/accel/kernel/finish.hpp"
#include "src/accel/kernel/grid.hpp"
#include "src/accel/kernel/prepared/model.hpp"
#include "src/accel/kernel/submission.hpp"
#include "src/accel/kernel/telemetry.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string_view>
#include <thread>

namespace node_accel_contract {
namespace {

using rund::node::accel::detail::KernelResult;

void IgnoreCompletion(void *, KernelResult) noexcept {}

struct SubmissionOwner final {};

[[nodiscard]] bool SubmissionTransitions() {
  using namespace rund::node::accel::detail;
  SubmissionOwner owner{};
  submission::State<SubmissionOwner> state{};
  std::atomic_bool start{};
  std::atomic_uint accepted{};
  const auto claim = [&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    if (submission::Begin(state, owner, IgnoreCompletion, nullptr)) {
      accepted.fetch_add(1u, std::memory_order_relaxed);
    }
  };
  std::thread first{claim};
  std::thread second{claim};
  start.store(true, std::memory_order_release);
  first.join();
  second.join();
  if (accepted.load(std::memory_order_relaxed) != 1u) {
    return false;
  }
  const submission::Claim<SubmissionOwner> taken = submission::Take(state);
  if (!taken || taken.owner != &owner || taken.completion != IgnoreCompletion ||
      taken.user != nullptr || submission::Take(state)) {
    return false;
  }
  if (submission::Begin(state, owner, nullptr, nullptr) ||
      !submission::Begin(state, owner, IgnoreCompletion, &owner)) {
    return false;
  }
  submission::Cancel(state);
  if (submission::Take(state) ||
      !submission::Begin(state, owner, IgnoreCompletion, nullptr)) {
    return false;
  }
  submission::Cancel(state);
  return true;
}

[[nodiscard]] bool PreparedPipelineClaimHasOneAuthority() {
  using namespace rund::node::accel::detail;
  prepared::PipelineSubmission submission{};
  if (submission.active() || submission.pipeline() != nullptr) {
    return false;
  }
  const auto owner = std::make_shared<prepared::PipelineState>();
  submission.owner = owner;
  if (!submission.active() || submission.pipeline() != owner.get()) {
    return false;
  }
  submission.owner.reset();
  return !submission.active() && submission.pipeline() == nullptr;
}

[[nodiscard]] bool GridBoundaries() {
  using rund::node::accel::detail::Grid;
  using rund::node::accel::detail::PlanGrid;
  constexpr std::uint64_t width = 256u;
  constexpr std::uint64_t x_limit = 4u;
  constexpr std::uint64_t y_limit = 3u;
  const Grid one = PlanGrid(1u, width, x_limit, y_limit);
  const Grid row = PlanGrid(width * x_limit, width, x_limit, y_limit);
  const Grid next = PlanGrid(width * x_limit + 1u, width, x_limit, y_limit);
  const Grid full =
      PlanGrid(width * x_limit * y_limit, width, x_limit, y_limit);
  const Grid excess =
      PlanGrid(width * x_limit * y_limit + 1u, width, x_limit, y_limit);
  const Grid overflow = PlanGrid(std::numeric_limits<std::uint64_t>::max(), 1u,
                                 std::numeric_limits<std::uint32_t>::max(),
                                 std::numeric_limits<std::uint32_t>::max());
  return one.x == 1u && one.y == 1u && row.x == x_limit && row.y == 1u &&
         next.x == x_limit && next.y == 2u && full.x == x_limit &&
         full.y == y_limit && !excess.valid() && !overflow.valid() &&
         !PlanGrid(0u, width, x_limit, y_limit).valid() &&
         !PlanGrid(1u, 0u, x_limit, y_limit).valid();
}

struct FinishEntry final {
  rund::AccelCheck result{};
};

struct FinishResources final {
  std::array<FinishEntry, 3u> entries{};

  [[nodiscard]] std::size_t size() const noexcept { return entries.size(); }
  [[nodiscard]] FinishEntry *entry(const std::size_t index) noexcept {
    return index < entries.size() ? &entries[index] : nullptr;
  }
};

[[nodiscard]] bool FinishPrecedence() {
  using rund::node::accel::detail::finish::Steps;
  FinishResources resources{.entries = {
                                FinishEntry{rund::AccelCheck{true, "ok"}},
                                FinishEntry{rund::AccelCheck{true, "ok"}},
                                FinishEntry{rund::AccelCheck{true, "ok"}},
                            }};
  resources.entries[0].result.failed_batches = 2u;
  resources.entries[0].result.first_failed_batch = 7u;
  resources.entries[0].result.first_status = 11u;
  resources.entries[1].result.failed_batches = 3u;
  resources.entries[1].result.first_failed_batch = 13u;
  resources.entries[1].result.first_status = 17u;
  const rund::AccelCheck folded =
      Steps(resources, [](const FinishEntry &entry) { return entry.result; });
  if (!folded.ok || folded.failed_batches != 5u ||
      folded.first_failed_batch != 7u || folded.first_status != 11u) {
    return false;
  }
  resources.entries[1].result = rund::AccelCheck{false, "step_failed"};
  const rund::AccelCheck failed =
      Steps(resources, [](const FinishEntry &entry) { return entry.result; });
  return !failed.ok && failed.reason != nullptr &&
         std::string_view{failed.reason} == "step_failed";
}

[[nodiscard]] bool TelemetryProjection() {
  using namespace rund::node::accel::detail;
  const PreparedPipelineControl control{
      .generated_item_count = 1u,
      .generated_capacity = 2u,
      .indirect_dispatch_count = 3u,
      .indirect_work_item_count = 4u,
      .iteration_count = 5u,
      .skipped_iteration_count = 6u,
      .conflict_count = 7u,
      .overflow_ordinal = 8u,
  };
  rund::RuntimeStats stats{};
  ProjectTelemetry(control, stats);
  return stats.generated_item_count == 1u && stats.generated_capacity == 2u &&
         stats.indirect_dispatch_count == 3u &&
         stats.indirect_work_item_count == 4u && stats.iteration_count == 5u &&
         stats.skipped_iteration_count == 6u && stats.conflict_count == 7u &&
         stats.overflow_ordinal == 8u;
}

} // namespace

bool AuthorityContract() {
  return SubmissionTransitions() && PreparedPipelineClaimHasOneAuthority() &&
         GridBoundaries() && FinishPrecedence() && TelemetryProjection();
}

} // namespace node_accel_contract
