#include "local.hpp"

#include <array>
#include <limits>

namespace rund_node_graph_services::memory_detail {

bool CheckCapacityMemoryPlan() {
  constexpr std::uint64_t MiB = 1ull << 20u;
  constexpr std::uint64_t oversize_bytes = (1ull << 30u) + 256u;
  Info oversize{};
  oversize.resources = {
      ResourceOf(1u, Value::U32, oversize_bytes / 4u, 4u),
  };
  const Resource &oversize_value = oversize.resources.front();
  oversize.nodes = {
      Node{.index = 0u,
           .operation = Operation::Map,
           .elements = oversize_value.elements,
           .accesses = {AccessOf(oversize_value, AccessMode::Write)}},
      Node{.index = 1u,
           .operation = Operation::Map,
           .elements = oversize_value.elements,
           .accesses = {AccessOf(oversize_value, AccessMode::Read)}},
  };
  const std::array<MemoryNode, 2u> oversize_writes{
      MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full},
  };
  Info oversize_rejected = oversize;
  const auto page_rejected = Plan(oversize_rejected, oversize_writes);
  if (page_rejected ||
      page_rejected.reason() != rund::compute::Reason::GraphCapacity ||
      !Plan(oversize, oversize_writes, WidePage) ||
      oversize.memory.logical_bytes != oversize_bytes ||
      oversize.memory.live_bytes != oversize_bytes ||
      oversize.memory.physical_bytes != oversize_bytes ||
      oversize.memory.allocation_count != 1u ||
      oversize.resources.front().alias_group != 1u ||
      oversize.resources.front().alias_offset_bytes != 0u ||
      oversize.resources.front().requires_reset()) {
    return false;
  }

  Info disjoint{};
  disjoint.resources = {
      ResourceOf(1u, Value::U32, oversize_bytes / 4u, 4u),
      ResourceOf(2u, Value::U32, oversize_bytes / 4u, 4u),
  };
  const Resource &first_large = disjoint.resources[0u];
  const Resource &second_large = disjoint.resources[1u];
  disjoint.nodes = {
      Node{.index = 0u,
           .operation = Operation::Map,
           .elements = first_large.elements,
           .accesses = {AccessOf(first_large, AccessMode::Write)}},
      Node{.index = 1u,
           .operation = Operation::Map,
           .elements = first_large.elements,
           .accesses = {AccessOf(first_large, AccessMode::Read)}},
      Node{.index = 2u,
           .operation = Operation::Map,
           .elements = second_large.elements,
           .accesses = {AccessOf(second_large, AccessMode::Write)}},
      Node{.index = 3u,
           .operation = Operation::Map,
           .elements = second_large.elements,
           .accesses = {AccessOf(second_large, AccessMode::Read)}},
  };
  const std::array<MemoryNode, 4u> disjoint_writes{
      MemoryNode{.write = Write::Full}, MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full}, MemoryNode{.write = Write::Full}};
  Info disjoint_repeat = disjoint;
  if (!Plan(disjoint, disjoint_writes, WidePage) ||
      !Plan(disjoint_repeat, disjoint_writes, WidePage) ||
      !SamePlan(disjoint, disjoint_repeat) ||
      disjoint.memory.logical_bytes != 2u * oversize_bytes ||
      disjoint.memory.live_bytes != oversize_bytes ||
      disjoint.memory.physical_bytes != oversize_bytes ||
      disjoint.memory.allocation_count != 1u ||
      disjoint.resources[0u].alias_group !=
          disjoint.resources[1u].alias_group ||
      disjoint.resources[0u].alias_offset_bytes != 0u ||
      disjoint.resources[1u].alias_offset_bytes != 0u) {
    return false;
  }

  constexpr std::uint64_t grown_bytes = oversize_bytes + MiB;
  Info grown{};
  grown.resources = {
      ResourceOf(1u, Value::U32, oversize_bytes / 4u, 4u),
      ResourceOf(2u, Value::U32, grown_bytes / 4u, 4u),
  };
  const Resource &smaller_large = grown.resources[0u];
  const Resource &larger_large = grown.resources[1u];
  grown.nodes = {
      Node{.index = 0u,
           .operation = Operation::Map,
           .elements = smaller_large.elements,
           .accesses = {AccessOf(smaller_large, AccessMode::Write)}},
      Node{.index = 1u,
           .operation = Operation::Map,
           .elements = smaller_large.elements,
           .accesses = {AccessOf(smaller_large, AccessMode::Read)}},
      Node{.index = 2u,
           .operation = Operation::Map,
           .elements = larger_large.elements,
           .accesses = {AccessOf(larger_large, AccessMode::Write)}},
      Node{.index = 3u,
           .operation = Operation::Map,
           .elements = larger_large.elements,
           .accesses = {AccessOf(larger_large, AccessMode::Read)}},
  };
  if (!Plan(grown, disjoint_writes, WidePage) ||
      grown.memory.logical_bytes != oversize_bytes + grown_bytes ||
      grown.memory.live_bytes != grown_bytes ||
      grown.memory.physical_bytes != grown_bytes ||
      grown.memory.allocation_count != 1u ||
      grown.resources[0u].alias_group != grown.resources[1u].alias_group ||
      grown.resources[0u].alias_offset_bytes != 0u ||
      grown.resources[1u].alias_offset_bytes != 0u) {
    return false;
  }

  Info overlap{};
  overlap.resources = {
      ResourceOf(1u, Value::U32, oversize_bytes / 4u, 4u),
      ResourceOf(2u, Value::U32, oversize_bytes / 4u, 4u),
  };
  const Resource &overlap_left = overlap.resources[0u];
  const Resource &overlap_right = overlap.resources[1u];
  overlap.nodes = {
      Node{.index = 0u,
           .operation = Operation::Map,
           .elements = overlap_left.elements,
           .accesses = {AccessOf(overlap_left, AccessMode::Write)}},
      Node{.index = 1u,
           .operation = Operation::Map,
           .elements = overlap_right.elements,
           .accesses = {AccessOf(overlap_right, AccessMode::Write)}},
      Node{.index = 2u,
           .operation = Operation::Map,
           .elements = overlap_left.elements,
           .accesses = {AccessOf(overlap_left, AccessMode::Read),
                        AccessOf(overlap_right, AccessMode::Read)}},
  };
  const std::array<MemoryNode, 3u> overlap_writes{
      MemoryNode{.write = Write::Full}, MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full}};
  if (!Plan(overlap, overlap_writes, WidePage) ||
      overlap.memory.logical_bytes != 2u * oversize_bytes ||
      overlap.memory.live_bytes != 2u * oversize_bytes ||
      overlap.memory.physical_bytes != 2u * oversize_bytes ||
      overlap.memory.allocation_count != 2u ||
      overlap.resources[0u].alias_group == overlap.resources[1u].alias_group ||
      Overlaps(overlap.resources[0u], overlap.resources[1u])) {
    return false;
  }

  Info destructive_large = overlap;
  destructive_large.nodes = {
      Node{.index = 0u,
           .operation = Operation::Map,
           .elements = overlap_left.elements,
           .accesses = {AccessOf(overlap_left, AccessMode::Write)}},
      Node{.index = 1u,
           .operation = Operation::Map,
           .elements = overlap_right.elements,
           .accesses = {AccessOf(overlap_left, AccessMode::Read),
                        AccessOf(overlap_right, AccessMode::Write)}},
      Node{.index = 2u,
           .operation = Operation::Map,
           .elements = overlap_right.elements,
           .accesses = {AccessOf(overlap_right, AccessMode::Read)}},
  };
  const std::array<MemoryNode, 3u> destructive_writes{
      MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full, .inplace = Inplace::Pointwise},
      MemoryNode{.write = Write::Full}};
  if (!Plan(destructive_large, destructive_writes, WidePage) ||
      destructive_large.memory.logical_bytes != 2u * oversize_bytes ||
      destructive_large.memory.live_bytes != 2u * oversize_bytes ||
      destructive_large.memory.physical_bytes != oversize_bytes ||
      destructive_large.memory.allocation_count != 1u ||
      destructive_large.resources[1u].source != 1u ||
      destructive_large.resources[0u].alias_group !=
          destructive_large.resources[1u].alias_group ||
      destructive_large.resources[0u].alias_offset_bytes != 0u ||
      destructive_large.resources[1u].alias_offset_bytes != 0u) {
    return false;
  }
  return true;
}

} // namespace rund_node_graph_services::memory_detail
