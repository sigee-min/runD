#include "../plan.hpp"

#include "internal.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::compute::detail::residency::execution {
namespace {

[[nodiscard]] bool use(const Materialization &materialization,
                       const std::uint64_t page, const std::uint64_t epoch,
                       CacheUse &result) noexcept {
  CacheKey key{};
  std::uint64_t page_offset = 0u;
  if (!project_graph_cache_key(
          materialization.cache,
          PageKey{.resource = materialization.cache.resource, .page = page},
          key) ||
      !plan_internal::checked_mul(page, materialization.payload_bytes,
                                  page_offset) ||
      page_offset >= materialization.logical_bytes) {
    return false;
  }
  std::uint64_t next_use = NeverUse;
  if (materialization.next_use_base != NeverUse) {
    std::uint64_t delta = 0u;
    if (!plan_internal::checked_mul(page, materialization.next_use_stride,
                                    delta) ||
        !plan_internal::checked_add(materialization.next_use_base, delta,
                                    next_use)) {
      next_use = NeverUse;
    }
  }
  DirtyExtent dirty{};
  if (writes(materialization.access)) {
    if (!plan_internal::checked_add(materialization.dirty_origin, page_offset,
                                    dirty.offset)) {
      return false;
    }
    dirty.bytes = std::min(materialization.payload_bytes,
                           materialization.logical_bytes - page_offset);
  }
  result = CacheUse{.key = key,
                    .access = materialization.access,
                    .next_use = next_use,
                    .dirty = dirty,
                    .epoch = epoch,
                    .retain_until = materialization.retain_until};
  return true;
}

} // namespace

bool Plan::project(const NodeId id, Node &result) const noexcept {
  result = {};
  if (identity_ == 0u || id.epoch >= epoch_count_ ||
      id.epoch >
          std::numeric_limits<std::uint64_t>::max() / request_.frame_capacity) {
    return false;
  }
  const std::uint64_t first = id.epoch * request_.frame_capacity;
  const std::uint32_t count = static_cast<std::uint32_t>(
      std::min(request_.page_count - first,
               static_cast<std::uint64_t>(request_.frame_capacity)));
  const std::uint32_t mask = count == UseCapacity
                                 ? std::numeric_limits<std::uint32_t>::max()
                                 : (std::uint32_t{1u} << count) - 1u;
  const std::size_t bank = static_cast<std::size_t>(id.epoch % BankCapacity);
  result.id = id;
  result.bank = static_cast<std::uint32_t>(bank);
  result.active_mask = mask;
  // Every phase may partially modify its authenticated target. A coherent
  // alias can elide physical transfer without eliding failure quarantine.
  result.may_write = true;
  if (id.phase == Phase::Input) {
    result.domain = Domain::HostService;
    result.route = Route{.source = request_.host_input[bank],
                         .target = request_.device_input[bank]};
    result.input_count = count;
    // Backing fill mutates the Host source; transfer or coherent execution can
    // then mutate Device Input. Both owners are exact quarantine targets.
    result.mutations[0] = request_.host_input[bank];
    result.mutations[1] = request_.device_input[bank];
    result.mutation_count = 2u;
  } else if (id.phase == Phase::Dispatch) {
    result.domain = Domain::Native;
    result.route = Route{.source = request_.device_input[bank],
                         .target = request_.device_output[bank]};
    result.input_count = count;
    result.output_count = count;
    result.mutations[0] = request_.device_output[bank];
    result.mutation_count = 1u;
  } else {
    result.domain = Domain::HostService;
    result.route = Route{.source = request_.device_output[bank],
                         .target = request_.host_output[bank]};
    result.output_count = count;
    result.mutations[0] = request_.host_output[bank];
    result.mutation_count = 1u;
  }
  for (std::size_t index = 0u; index < result.input_count; ++index) {
    if (!use(request_.input, first + index, id.epoch, result.input[index])) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < result.output_count; ++index) {
    if (!use(request_.output, first + index, id.epoch, result.output[index])) {
      return false;
    }
  }
  return true;
}

bool Plan::forecast(const std::uint64_t epoch,
                    Forecast &result) const noexcept {
  result = {};
  if (identity_ == 0u || epoch >= epoch_count_ ||
      epoch >
          std::numeric_limits<std::uint64_t>::max() / request_.frame_capacity) {
    return false;
  }
  const std::uint64_t first = epoch * request_.frame_capacity;
  result = Forecast{
      .epoch = epoch,
      .first_page = first,
      .page_count =
          std::min(request_.page_count - first,
                   static_cast<std::uint64_t>(request_.frame_capacity)),
      .prefetch_epoch = epoch > request_.prefetch_distance
                            ? epoch - request_.prefetch_distance
                            : 0u,
      .ready_epoch = epoch,
  };
  return result.page_count != 0u;
}

bool Plan::predecessors(const NodeId id,
                        const std::span<NodeId, PredecessorCapacity> storage,
                        std::size_t &count) const noexcept {
  count = 0u;
  Node projected{};
  if (storage.size() < PredecessorCapacity || !project(id, projected)) {
    return false;
  }
  if (id.phase == Phase::Input) {
    if (id.epoch >= BankCapacity) {
      storage[count++] =
          NodeId{.epoch = id.epoch - BankCapacity, .phase = Phase::Dispatch};
    }
    return true;
  }
  if (id.phase == Phase::Dispatch) {
    storage[count++] = NodeId{.epoch = id.epoch, .phase = Phase::Input};
    if (id.epoch >= BankCapacity) {
      storage[count++] =
          NodeId{.epoch = id.epoch - BankCapacity, .phase = Phase::Output};
    }
    return true;
  }
  storage[count++] = NodeId{.epoch = id.epoch, .phase = Phase::Dispatch};
  return true;
}

} // namespace rund::compute::detail::residency::execution
