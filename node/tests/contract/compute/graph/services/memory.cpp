#include "model.hpp"

#include "../../../../../src/compute/resource/memory.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

namespace rund_node_graph_services {

[[nodiscard]] bool ValidMemoryPlan() {
  using rund::compute::graph::Access;
  using rund::compute::graph::Info;
  using rund::compute::graph::Node;
  using rund::compute::graph::Resource;
  using rund::compute::graph::Value;

  const auto resource = [](const std::uint32_t id, const Value type,
                           const std::uint64_t elements,
                           const std::uint64_t width) {
    return Resource{.id = id,
                    .type = type,
                    .visibility = Visibility::Internal,
                    .elements = elements,
                    .element_bytes = width,
                    .bytes = elements * width};
  };
  const auto access = [](const Resource &value, const AccessMode mode) {
    return Access{.resource = value.id,
                  .mode = mode,
                  .size_bytes = value.bytes,
                  .element_bytes = value.element_bytes,
                  .element_count = value.elements,
                  .stride_bytes = value.element_bytes};
  };
  const auto overlaps = [](const Resource &left, const Resource &right) {
    return left.alias_group == right.alias_group &&
           left.alias_offset_bytes < right.alias_offset_bytes + right.bytes &&
           right.alias_offset_bytes < left.alias_offset_bytes + left.bytes;
  };
  const auto same_plan = [](const Info &left, const Info &right) {
    if (left.memory != right.memory ||
        left.resources.size() != right.resources.size()) {
      return false;
    }
    for (std::size_t index = 0u; index < left.resources.size(); ++index) {
      const Resource &first = left.resources[index];
      const Resource &second = right.resources[index];
      if (first.active != second.active || first.parent != second.parent ||
          first.source != second.source ||
          first.alias_group != second.alias_group ||
          first.alias_offset_bytes != second.alias_offset_bytes ||
          first.reset_node != second.reset_node) {
        return false;
      }
    }
    return true;
  };

  Info authored{};
  authored.resources = {
      resource(1u, Value::U32, 128u, 4u),
      resource(2u, Value::U64, 32u, 8u),
      resource(3u, Value::U32, 32u, 4u),
  };
  const Resource &wide32 = authored.resources[0u];
  const Resource &wide64 = authored.resources[1u];
  const Resource &small32 = authored.resources[2u];
  authored.nodes = {
      Node{.index = 0u,
           .operation = Operation::Map,
           .elements = wide32.elements,
           .accesses = {access(wide32, AccessMode::Write)}},
      Node{.index = 1u,
           .operation = Operation::Map,
           .elements = wide32.elements,
           .accesses = {access(wide32, AccessMode::Read),
                        access(wide64, AccessMode::Write)}},
      Node{.index = 2u,
           .operation = Operation::Map,
           .elements = wide64.elements,
           .accesses = {access(wide64, AccessMode::Read),
                        access(small32, AccessMode::Write)}},
      Node{.index = 3u,
           .operation = Operation::Map,
           .elements = small32.elements,
           .accesses = {access(small32, AccessMode::Read)}},
  };
  using rund::compute::detail::resource_detail::Inplace;
  using rund::compute::detail::resource_detail::MemoryNode;
  using rund::compute::detail::resource_detail::Write;
  constexpr std::uint64_t Page = 1ull << 30u;
  constexpr std::uint64_t WidePage = 4ull << 30u;
  const auto plan = [](Info &info, const std::span<const MemoryNode> nodes,
                       const std::uint64_t page = Page) {
    return rund::compute::detail::resource_detail::plan_memory(info, nodes,
                                                               page);
  };
  const std::array<MemoryNode, 4u> complete{
      MemoryNode{.write = Write::Full}, MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full}, MemoryNode{.write = Write::Full}};
  Info planned = authored;
  Info repeated = authored;
  const auto first = plan(planned, complete);
  const auto second = plan(repeated, complete);
  if (!first || !second || !same_plan(planned, repeated) ||
      planned.memory.logical_bytes != 896u ||
      planned.memory.live_bytes != 768u ||
      planned.memory.physical_bytes != 768u ||
      planned.memory.allocation_count != 1u) {
    return false;
  }
  const Resource &first32 = planned.resources[0u];
  const Resource &first64 = planned.resources[1u];
  const Resource &last32 = planned.resources[2u];
  if (first32.alias_group != first64.alias_group ||
      first32.alias_group != last32.alias_group ||
      first32.alias_offset_bytes != last32.alias_offset_bytes ||
      first32.bytes == last32.bytes || overlaps(first32, first64) ||
      overlaps(first64, last32) || first32.type != Value::U32 ||
      first64.type != Value::U64) {
    return false;
  }
  for (const Resource &value : planned.resources) {
    if (value.alias_offset_bytes % 256u != 0u || value.requires_reset()) {
      return false;
    }
  }

  Info reset{};
  reset.resources = {
      resource(1u, Value::U32, 16u, 4u),
      resource(2u, Value::U64, 4u, 8u),
  };
  const Resource &reset32 = reset.resources[0u];
  const Resource &reset64 = reset.resources[1u];
  reset.nodes = {
      Node{.index = 0u,
           .operation = Operation::Compact,
           .elements = reset32.elements,
           .accesses = {access(reset32, AccessMode::Write)}},
      Node{.index = 1u,
           .operation = Operation::Map,
           .elements = reset32.elements,
           .accesses = {access(reset32, AccessMode::Read)}},
      Node{.index = 2u,
           .operation = Operation::Compact,
           .elements = reset64.elements,
           .accesses = {access(reset64, AccessMode::Write)}},
      Node{.index = 3u,
           .operation = Operation::Map,
           .elements = reset64.elements,
           .accesses = {access(reset64, AccessMode::Read)}},
  };
  const std::array<MemoryNode, 4u> partial{
      MemoryNode{.write = Write::Partial}, MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Partial}, MemoryNode{.write = Write::Full}};
  if (!plan(reset, partial) || reset.memory.logical_bytes != 96u ||
      reset.memory.live_bytes != 64u || reset.memory.physical_bytes != 64u ||
      reset.memory.allocation_count != 1u || reset.memory.reset_bytes != 96u ||
      reset.memory.reset_count != 2u) {
    return false;
  }
  const Resource &stored32 = reset.resources[0u];
  const Resource &stored64 = reset.resources[1u];
  if (!stored32.requires_reset() || !stored64.requires_reset() ||
      stored32.alias_group != stored64.alias_group ||
      stored32.alias_offset_bytes != 0u || stored64.alias_offset_bytes != 0u ||
      !overlaps(stored32, stored64)) {
    return false;
  }

  constexpr std::uint64_t MiB = 1ull << 20u;
  constexpr std::uint64_t large_bytes = 700u * MiB;
  Info chunked{};
  chunked.resources = {
      resource(1u, Value::U32, large_bytes / 4u, 4u),
      resource(2u, Value::U32, large_bytes / 4u, 4u),
  };
  const Resource &chunk_left = chunked.resources[0u];
  const Resource &chunk_right = chunked.resources[1u];
  chunked.nodes = {
      Node{.index = 0u,
           .operation = Operation::Map,
           .elements = chunk_left.elements,
           .accesses = {access(chunk_left, AccessMode::Write)}},
      Node{.index = 1u,
           .operation = Operation::Map,
           .elements = chunk_right.elements,
           .accesses = {access(chunk_right, AccessMode::Write)}},
      Node{.index = 2u,
           .operation = Operation::Map,
           .elements = chunk_left.elements,
           .accesses = {access(chunk_left, AccessMode::Read),
                        access(chunk_right, AccessMode::Read)}},
  };
  const std::array<MemoryNode, 3u> chunk_writes{
      MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full},
  };
  if (!plan(chunked, chunk_writes, WidePage) ||
      chunked.memory.logical_bytes != 2u * large_bytes ||
      chunked.memory.live_bytes != 2u * large_bytes ||
      chunked.memory.physical_bytes != 2u * large_bytes ||
      chunked.memory.allocation_count != 2u ||
      chunked.resources[0u].alias_group == chunked.resources[1u].alias_group ||
      chunked.resources[0u].alias_offset_bytes != 0u ||
      chunked.resources[1u].alias_offset_bytes != 0u) {
    return false;
  }

  Info multi_source{};
  multi_source.resources = {
      resource(1u, Value::U32, 128u, 4u),
      resource(2u, Value::U32, 128u, 4u),
      resource(3u, Value::U32, 128u, 4u),
  };
  const Resource &left_source = multi_source.resources[0u];
  const Resource &right_source = multi_source.resources[1u];
  const Resource &map_output = multi_source.resources[2u];
  multi_source.nodes = {
      Node{.index = 0u,
           .operation = Operation::Map,
           .elements = left_source.elements,
           .accesses = {access(left_source, AccessMode::Write)}},
      Node{.index = 1u,
           .operation = Operation::Map,
           .elements = right_source.elements,
           .accesses = {access(right_source, AccessMode::Write)}},
      Node{.index = 2u,
           .operation = Operation::Map,
           .elements = map_output.elements,
           .accesses = {access(left_source, AccessMode::Read),
                        access(right_source, AccessMode::Read),
                        access(map_output, AccessMode::Write)}},
      Node{.index = 3u,
           .operation = Operation::Map,
           .elements = map_output.elements,
           .accesses = {access(map_output, AccessMode::Read)}},
  };
  const std::array<MemoryNode, 4u> multi_source_writes{
      MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full, .inplace = Inplace::Pointwise},
      MemoryNode{.write = Write::Full},
  };
  Info indexed_source = multi_source;
  const std::array<MemoryNode, 4u> indexed_writes{
      MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full, .inplace = Inplace::None},
      MemoryNode{.write = Write::Full},
  };
  if (!plan(multi_source, multi_source_writes) ||
      multi_source.memory.logical_bytes != 1536u ||
      multi_source.memory.live_bytes != 1536u ||
      multi_source.memory.physical_bytes != 1024u ||
      multi_source.memory.allocation_count != 1u ||
      multi_source.resources[2u].source != 1u ||
      multi_source.resources[2u].alias_group !=
          multi_source.resources[0u].alias_group ||
      multi_source.resources[2u].alias_offset_bytes !=
          multi_source.resources[0u].alias_offset_bytes ||
      overlaps(multi_source.resources[0u], multi_source.resources[1u])) {
    return false;
  }
  if (!plan(indexed_source, indexed_writes) ||
      indexed_source.memory.logical_bytes != 1536u ||
      indexed_source.memory.live_bytes != 1536u ||
      indexed_source.memory.physical_bytes != 1536u ||
      indexed_source.memory.allocation_count != 1u ||
      indexed_source.resources[2u].source != 0u ||
      overlaps(indexed_source.resources[0u], indexed_source.resources[2u])) {
    return false;
  }

  constexpr std::uint64_t oversize_bytes = (1ull << 30u) + 256u;
  Info oversize{};
  oversize.resources = {
      resource(1u, Value::U32, oversize_bytes / 4u, 4u),
  };
  const Resource &oversize_value = oversize.resources.front();
  oversize.nodes = {
      Node{.index = 0u,
           .operation = Operation::Map,
           .elements = oversize_value.elements,
           .accesses = {access(oversize_value, AccessMode::Write)}},
      Node{.index = 1u,
           .operation = Operation::Map,
           .elements = oversize_value.elements,
           .accesses = {access(oversize_value, AccessMode::Read)}},
  };
  const std::array<MemoryNode, 2u> oversize_writes{
      MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full},
  };
  Info oversize_rejected = oversize;
  const auto page_rejected = plan(oversize_rejected, oversize_writes);
  if (page_rejected ||
      page_rejected.reason() != rund::compute::Reason::GraphCapacity ||
      !plan(oversize, oversize_writes, WidePage) ||
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
      resource(1u, Value::U32, oversize_bytes / 4u, 4u),
      resource(2u, Value::U32, oversize_bytes / 4u, 4u),
  };
  const Resource &first_large = disjoint.resources[0u];
  const Resource &second_large = disjoint.resources[1u];
  disjoint.nodes = {
      Node{.index = 0u,
           .operation = Operation::Map,
           .elements = first_large.elements,
           .accesses = {access(first_large, AccessMode::Write)}},
      Node{.index = 1u,
           .operation = Operation::Map,
           .elements = first_large.elements,
           .accesses = {access(first_large, AccessMode::Read)}},
      Node{.index = 2u,
           .operation = Operation::Map,
           .elements = second_large.elements,
           .accesses = {access(second_large, AccessMode::Write)}},
      Node{.index = 3u,
           .operation = Operation::Map,
           .elements = second_large.elements,
           .accesses = {access(second_large, AccessMode::Read)}},
  };
  const std::array<MemoryNode, 4u> disjoint_writes{
      MemoryNode{.write = Write::Full}, MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full}, MemoryNode{.write = Write::Full}};
  Info disjoint_repeat = disjoint;
  if (!plan(disjoint, disjoint_writes, WidePage) ||
      !plan(disjoint_repeat, disjoint_writes, WidePage) ||
      !same_plan(disjoint, disjoint_repeat) ||
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
      resource(1u, Value::U32, oversize_bytes / 4u, 4u),
      resource(2u, Value::U32, grown_bytes / 4u, 4u),
  };
  const Resource &smaller_large = grown.resources[0u];
  const Resource &larger_large = grown.resources[1u];
  grown.nodes = {
      Node{.index = 0u,
           .operation = Operation::Map,
           .elements = smaller_large.elements,
           .accesses = {access(smaller_large, AccessMode::Write)}},
      Node{.index = 1u,
           .operation = Operation::Map,
           .elements = smaller_large.elements,
           .accesses = {access(smaller_large, AccessMode::Read)}},
      Node{.index = 2u,
           .operation = Operation::Map,
           .elements = larger_large.elements,
           .accesses = {access(larger_large, AccessMode::Write)}},
      Node{.index = 3u,
           .operation = Operation::Map,
           .elements = larger_large.elements,
           .accesses = {access(larger_large, AccessMode::Read)}},
  };
  if (!plan(grown, disjoint_writes, WidePage) ||
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
      resource(1u, Value::U32, oversize_bytes / 4u, 4u),
      resource(2u, Value::U32, oversize_bytes / 4u, 4u),
  };
  const Resource &overlap_left = overlap.resources[0u];
  const Resource &overlap_right = overlap.resources[1u];
  overlap.nodes = {
      Node{.index = 0u,
           .operation = Operation::Map,
           .elements = overlap_left.elements,
           .accesses = {access(overlap_left, AccessMode::Write)}},
      Node{.index = 1u,
           .operation = Operation::Map,
           .elements = overlap_right.elements,
           .accesses = {access(overlap_right, AccessMode::Write)}},
      Node{.index = 2u,
           .operation = Operation::Map,
           .elements = overlap_left.elements,
           .accesses = {access(overlap_left, AccessMode::Read),
                        access(overlap_right, AccessMode::Read)}},
  };
  const std::array<MemoryNode, 3u> overlap_writes{
      MemoryNode{.write = Write::Full}, MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full}};
  if (!plan(overlap, overlap_writes, WidePage) ||
      overlap.memory.logical_bytes != 2u * oversize_bytes ||
      overlap.memory.live_bytes != 2u * oversize_bytes ||
      overlap.memory.physical_bytes != 2u * oversize_bytes ||
      overlap.memory.allocation_count != 2u ||
      overlap.resources[0u].alias_group == overlap.resources[1u].alias_group ||
      overlaps(overlap.resources[0u], overlap.resources[1u])) {
    return false;
  }

  Info destructive_large = overlap;
  destructive_large.nodes = {
      Node{.index = 0u,
           .operation = Operation::Map,
           .elements = overlap_left.elements,
           .accesses = {access(overlap_left, AccessMode::Write)}},
      Node{.index = 1u,
           .operation = Operation::Map,
           .elements = overlap_right.elements,
           .accesses = {access(overlap_left, AccessMode::Read),
                        access(overlap_right, AccessMode::Write)}},
      Node{.index = 2u,
           .operation = Operation::Map,
           .elements = overlap_right.elements,
           .accesses = {access(overlap_right, AccessMode::Read)}},
  };
  const std::array<MemoryNode, 3u> destructive_writes{
      MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Full, .inplace = Inplace::Pointwise},
      MemoryNode{.write = Write::Full}};
  if (!plan(destructive_large, destructive_writes, WidePage) ||
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

  Info read_before_reset{};
  read_before_reset.resources = {
      resource(1u, Value::U32, 4u, 4u),
  };
  read_before_reset.resources.front().visibility = Visibility::Output;
  const Resource &late_output = read_before_reset.resources.front();
  read_before_reset.nodes = {
      Node{.index = 0u,
           .operation = Operation::Map,
           .elements = late_output.elements,
           .accesses = {access(late_output, AccessMode::Read)}},
      Node{.index = 1u,
           .operation = Operation::Compact,
           .elements = late_output.elements,
           .accesses = {access(late_output, AccessMode::Write)}},
  };
  const std::array<MemoryNode, 2u> late_writes{
      MemoryNode{.write = Write::Full},
      MemoryNode{.write = Write::Partial},
  };
  const auto late_rejected = plan(read_before_reset, late_writes);
  if (late_rejected ||
      late_rejected.reason() != rund::compute::Reason::GraphInvalid) {
    return false;
  }

  constexpr std::uint64_t widest =
      std::numeric_limits<std::uint64_t>::max() / 4u;
  Info overflow{};
  overflow.resources = {
      resource(1u, Value::U32, widest, 4u),
      resource(2u, Value::U32, widest, 4u),
  };
  const Resource &overflow_left = overflow.resources[0u];
  const Resource &overflow_right = overflow.resources[1u];
  overflow.nodes = {
      Node{.index = 0u,
           .operation = Operation::Map,
           .elements = widest,
           .accesses = {access(overflow_left, AccessMode::Write),
                        access(overflow_right, AccessMode::Write)}},
  };
  const std::array<MemoryNode, 1u> overflow_writes{
      MemoryNode{.write = Write::Full}};
  const auto rejected = plan(overflow, overflow_writes, WidePage);
  return !rejected && rejected.reason() == rund::compute::Reason::GraphCapacity;
}

} // namespace rund_node_graph_services
