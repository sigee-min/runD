#include "local.hpp"

#include <algorithm>
#include <cstdint>
#include <tuple>

namespace rund::compute::detail::resource_detail::memory_detail {

Status finalize(graph::Info &info, const std::uint64_t page_bytes, Work &work) {
  std::uint64_t live_bytes = 0u;
  for (std::size_t node = 0u; node < info.nodes.size(); ++node) {
    if (node != 0u) {
      if (work.live[node - 1u].stop > live_bytes) {
        return Status::fail(Reason::GraphInvalid);
      }
      live_bytes -= work.live[node - 1u].stop;
    }
    if (!kernel::checked::add(live_bytes, work.live[node].start, live_bytes)) {
      return Status::fail(Reason::GraphCapacity);
    }
    info.memory.live_bytes = std::max(info.memory.live_bytes, live_bytes);
  }

  if (work.arena.empty()) {
    return Status::success();
  }

  std::sort(work.arena.begin(), work.arena.end(),
            [&](const std::uint32_t left, const std::uint32_t right) {
              const Lifetime left_lifetime = work.lifetimes[left - 1u];
              const Lifetime right_lifetime = work.lifetimes[right - 1u];
              return std::tie(left_lifetime.first, left) <
                     std::tie(right_lifetime.first, right);
            });
  Layout destructive;
  Layout ordinary;
  if (!pack(info.resources, work.lifetimes, work.arena, page_bytes, true,
            destructive) ||
      !pack(info.resources, work.lifetimes, work.arena, page_bytes, false,
            ordinary)) {
    return Status::fail(Reason::GraphCapacity);
  }
  const bool destructive_layout =
      std::tie(destructive.bytes, destructive.count) <=
      std::tie(ordinary.bytes, ordinary.count);
  const Layout &layout = destructive_layout ? destructive : ordinary;
  for (const std::uint32_t id : work.arena) {
    graph::Resource &value = info.resources[id - 1u];
    if (layout.owners[id - 1u] == 0u) {
      return Status::fail(Reason::GraphInvalid);
    }
    value.alias_group = layout.owners[id - 1u];
    value.alias_offset_bytes = layout.offsets[id - 1u];
    if (!destructive_layout) {
      value.source = 0u;
    }
  }
  if (!kernel::checked::add(info.memory.physical_bytes, layout.bytes,
                            info.memory.physical_bytes) ||
      !kernel::checked::add(info.memory.allocation_count, layout.count,
                            info.memory.allocation_count)) {
    return Status::fail(Reason::GraphCapacity);
  }
  return Status::success();
}

} // namespace rund::compute::detail::resource_detail::memory_detail
