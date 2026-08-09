#include "src/accel/kernel/backend/exception.hpp"
#include "src/accel/kernel/finish.hpp"
#include "src/accel/kernel/grid.hpp"
#include "src/accel/kernel/telemetry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>

#include "backend.hpp"

namespace node_accel_contract {

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
  return stats.run.work.generated_item_count == 1u &&
         stats.run.work.generated_capacity == 2u &&
         stats.run.work.indirect_dispatch_count == 3u &&
         stats.run.work.indirect_work_item_count == 4u &&
         stats.run.work.iteration_count == 5u &&
         stats.run.work.skipped_iteration_count == 6u &&
         stats.run.work.conflict_count == 7u &&
         stats.run.work.overflow_ordinal == 8u;
}

[[nodiscard]] bool BackendCapacityExceptionClassesAreCanonical() {
  using rund::node::accel::detail::backend_exception::
      RethrowUnlessCapacityException;
  const auto accepts = [](const auto &raise) {
    try {
      raise();
    } catch (...) {
      try {
        RethrowUnlessCapacityException();
        return true;
      } catch (...) {
        return false;
      }
    }
    return false;
  };
  return accepts([] { throw std::bad_alloc{}; }) &&
         accepts([] { throw std::length_error{"capacity"}; }) &&
         !accepts([] { throw 7; });
}

} // namespace node_accel_contract
