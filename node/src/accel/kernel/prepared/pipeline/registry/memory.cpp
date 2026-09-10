#include "../registry.hpp"

#include "../reservation.hpp"
#include "internal.hpp"

#include <kernel/core/checked.hpp>
#include <rund/counter.hpp>

#include <cstdint>
#include <limits>
#include <mutex>

namespace rund::node::accel::detail {

using ::rund::kernel::checked::mul;

PreparedMemory ReadPreparedKernelTemplateRegistryMemory(
    const PreparedKernelTemplateRegistry &registry) noexcept {
  PreparedKernelTemplateRegistryState *const state = registry_state(registry);
  if (state == nullptr) {
    return {};
  }
  const std::lock_guard<std::recursive_mutex> lock{state->mutex};
  std::uint64_t entries = 0u;
  std::uint64_t charges = 0u;
  std::uint64_t bytes = sizeof(PreparedKernelTemplateRegistryState);
  if (!mul(state->entries.capacity(), sizeof(PreparedKernelTemplateEntry),
           entries) ||
      !mul(state->template_charges.capacity(),
           sizeof(PreparedKernelTemplateCharge), charges) ||
      !accumulate(bytes, entries) || !accumulate(bytes, charges)) {
    bytes = std::numeric_limits<std::uint64_t>::max();
  }
  PreparedMemory memory{
      .current = bytes, .peak = bytes, .cumulative = bytes, .budget = bytes};
  const auto add_memory = [&](const PreparedMemory value) noexcept {
    memory.current =
        ::rund::detail::counter::SaturatingAdd(memory.current, value.current);
    memory.peak =
        ::rund::detail::counter::SaturatingAdd(memory.peak, value.peak);
    memory.cumulative = ::rund::detail::counter::SaturatingAdd(
        memory.cumulative, value.cumulative);
    memory.reused =
        ::rund::detail::counter::SaturatingAdd(memory.reused, value.reused);
    memory.budget =
        ::rund::detail::counter::SaturatingAdd(memory.budget, value.budget);
  };
  for (std::size_t index = 0u; index < state->entries.size(); ++index) {
    const PreparedKernelTemplateEntry &entry = state->entries[index];
    if (entry.prepared == nullptr || entry.ops == nullptr ||
        entry.ops->observe_pipeline_template == nullptr) {
      add_memory(PreparedMemory{
          .current = std::numeric_limits<std::uint64_t>::max(),
          .peak = std::numeric_limits<std::uint64_t>::max(),
          .cumulative = std::numeric_limits<std::uint64_t>::max(),
          .budget = std::numeric_limits<std::uint64_t>::max(),
      });
      break;
    }
    bool first_owner = true;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (state->entries[prior].prepared.get() == entry.prepared.get()) {
        first_owner = false;
        break;
      }
    }
    if (first_owner) {
      add_memory(entry.ops->observe_pipeline_template(entry.prepared.get()));
    }
  }
  return memory;
}

} // namespace rund::node::accel::detail
