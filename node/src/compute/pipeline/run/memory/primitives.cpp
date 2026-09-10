#include "internal.hpp"

#include "../../../device/residency/pool.hpp"

#include <algorithm>

namespace rund::compute::detail {

MemoryCounter pipeline_prepared_memory(
    const node::accel::detail::PreparedMemory memory) noexcept {
  return MemoryCounter{.current = memory.current,
                       .peak = memory.peak,
                       .cumulative = memory.cumulative,
                       .reused = memory.reused,
                       .budget = memory.budget};
}

MemoryCounter pipeline_remaining_memory(const MemoryCounter total,
                                        const MemoryCounter part) noexcept {
  return MemoryCounter{
      .current =
          ::rund::detail::counter::Remaining(total.current, part.current),
      .peak = ::rund::detail::counter::Remaining(total.peak, part.peak),
      .cumulative =
          ::rund::detail::counter::Remaining(total.cumulative, part.cumulative),
      .reused = ::rund::detail::counter::Remaining(total.reused, part.reused),
      .budget = ::rund::detail::counter::Remaining(total.budget, part.budget),
  };
}

bool pipeline_pool_owns_buffer(
    const residency::Pool *const pool,
    const std::shared_ptr<BufferState> &buffer) noexcept {
  if (pool == nullptr || buffer == nullptr) {
    return false;
  }
  const auto owns = [&buffer](const auto &owners) noexcept {
    return std::find(owners.begin(), owners.end(), buffer) != owners.end();
  };
  if (owns(pool->input) || owns(pool->intermediate) || owns(pool->control) ||
      owns(pool->output)) {
    return true;
  }
  return std::any_of(pool->graph_owners.begin(), pool->graph_owners.end(),
                     [&owns](const residency::PoolPhysicalOwner &owner) {
                       return owns(owner.buffers);
                     });
}

} // namespace rund::compute::detail
