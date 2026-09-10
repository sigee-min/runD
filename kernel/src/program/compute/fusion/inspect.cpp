#include "local.hpp"

#include <algorithm>
#include <cstddef>

namespace rund::kernel::fusion_detail {
namespace {

[[nodiscard]] BoundaryShape InspectBoundary(const Graph &graph,
                                            const u64 left_index) noexcept {
  const GraphNode &left = graph.nodes[left_index];
  const GraphNode &right = graph.nodes[left_index + 1u];
  BoundaryShape shape{};
  for (u64 index = 0u; index < left.buffer_count; ++index) {
    const GraphBufferRef &buffer = left.buffers[index];
    if (buffer.role == BufferRole::Write) {
      ++shape.producer_writes;
      if (shape.producer_writes == 1u) {
        shape.intermediate = buffer.logical_id;
      }
    }
  }
  if (shape.producer_writes == 1u) {
    u64 read_ordinal = 0u;
    for (u64 index = 0u; index < right.buffer_count; ++index) {
      const GraphBufferRef &buffer = right.buffers[index];
      if (buffer.role == BufferRole::Read) {
        if (buffer.logical_id == shape.intermediate) {
          shape.candidate = true;
          shape.consumer_read_ordinal = read_ordinal;
          ++shape.consumer_reads;
        }
        ++read_ordinal;
      }
    }
    return shape;
  }

  // A multiple-write producer is not fusible, but it remains a rejected
  // candidate when any produced value crosses the boundary. The first match
  // is the deterministic rejection identity.
  for (u64 left_buffer = 0u; left_buffer < left.buffer_count; ++left_buffer) {
    const GraphBufferRef &producer = left.buffers[left_buffer];
    if (producer.role != BufferRole::Write) {
      continue;
    }
    for (u64 right_buffer = 0u; right_buffer < right.buffer_count;
         ++right_buffer) {
      const GraphBufferRef &consumer = right.buffers[right_buffer];
      if (consumer.role == BufferRole::Read &&
          consumer.logical_id == producer.logical_id) {
        shape.candidate = true;
        shape.consumer_reads = 1u;
        shape.intermediate = producer.logical_id;
        return shape;
      }
    }
  }
  return shape;
}

[[nodiscard]] u64 LowerBoundId(const ReaderFact *const facts, const u64 count,
                               const u64 logical_id) noexcept {
  u64 first = 0u;
  u64 length = count;
  while (length != 0u) {
    const u64 half = length / 2u;
    const u64 middle = first + half;
    if (facts[middle].logical_id < logical_id) {
      first = middle + 1u;
      length -= half + 1u;
    } else {
      length = half;
    }
  }
  return first;
}

[[nodiscard]] u64 UpperBoundId(const ReaderFact *const facts, const u64 count,
                               const u64 first, const u64 logical_id) noexcept {
  u64 begin = first;
  u64 length = count - first;
  while (length != 0u) {
    const u64 half = length / 2u;
    const u64 middle = begin + half;
    if (facts[middle].logical_id <= logical_id) {
      begin = middle + 1u;
      length -= half + 1u;
    } else {
      length = half;
    }
  }
  return begin;
}

[[nodiscard]] u64 FindWriter(const ReaderFact *const facts, const u64 first,
                             const u64 last, const u64 writer) noexcept {
  u64 begin = first;
  u64 length = last - first;
  while (length != 0u) {
    const u64 half = length / 2u;
    const u64 middle = begin + half;
    if (facts[middle].writer < writer) {
      begin = middle + 1u;
      length -= half + 1u;
    } else {
      length = half;
    }
  }
  return begin < last && facts[begin].writer == writer ? begin : kNoCandidate;
}

} // namespace

u64 BuildReaderFacts(const Graph &graph, std::vector<BoundaryShape> &shapes,
                     std::vector<ReaderFact> &facts) noexcept {
  u64 count = 0u;
  for (u64 boundary = 0u; boundary + 1u < graph.node_count; ++boundary) {
    BoundaryShape &shape = shapes[boundary];
    shape = InspectBoundary(graph, boundary);
    if (!shape.candidate || shape.producer_writes != 1u ||
        shape.consumer_reads != 1u) {
      continue;
    }
    facts[count++] =
        ReaderFact{.logical_id = shape.intermediate, .writer = boundary};
  }
  std::sort(facts.begin(), facts.begin() + static_cast<std::ptrdiff_t>(count),
            [](const ReaderFact &left, const ReaderFact &right) noexcept {
              return left.logical_id < right.logical_id ||
                     (left.logical_id == right.logical_id &&
                      left.writer < right.writer);
            });

  for (u64 node_index = 0u; node_index < graph.node_count; ++node_index) {
    const GraphNode &node = graph.nodes[node_index];
    for (u64 buffer_index = 0u; buffer_index < node.buffer_count;
         ++buffer_index) {
      const GraphBufferRef &buffer = node.buffers[buffer_index];
      if (buffer.role != BufferRole::Read) {
        continue;
      }
      const u64 first = LowerBoundId(facts.data(), count, buffer.logical_id);
      if (first < count && facts[first].logical_id == buffer.logical_id &&
          facts[first].active != kNoCandidate) {
        ++facts[facts[first].active].readers;
      }
    }
    for (u64 buffer_index = 0u; buffer_index < node.buffer_count;
         ++buffer_index) {
      const GraphBufferRef &buffer = node.buffers[buffer_index];
      if (buffer.role != BufferRole::Write) {
        continue;
      }
      const u64 first = LowerBoundId(facts.data(), count, buffer.logical_id);
      if (first == count || facts[first].logical_id != buffer.logical_id) {
        continue;
      }
      const u64 last =
          UpperBoundId(facts.data(), count, first, buffer.logical_id);
      facts[first].active = FindWriter(facts.data(), first, last, node_index);
    }
  }
  return count;
}

u64 ReaderCount(const ReaderFact *const facts, const u64 count,
                const u64 logical_id, const u64 writer) noexcept {
  const u64 first = LowerBoundId(facts, count, logical_id);
  if (first == count || facts[first].logical_id != logical_id) {
    return 0u;
  }
  const u64 last = UpperBoundId(facts, count, first, logical_id);
  const u64 found = FindWriter(facts, first, last, writer);
  return found == kNoCandidate ? 0u : facts[found].readers;
}

} // namespace rund::kernel::fusion_detail
