#include "order.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <vector>

namespace rund::kernel::workspace_detail {
namespace {

struct U64KeyRange {
  u64 min_key = 0u;
  u64 max_key = 0u;
};

template <typename KeyFn>
bool StableCountingOrderByU32(Workspace& workspace,
                              const u32 packet_count,
                              KeyFn key_at) {
  if (packet_count == 0u) {
    return true;
  }
  u32 min_key = key_at(workspace.ordered_packet_indices[0u]);
  u32 max_key = min_key;
  for (u32 index = 1u; index < packet_count; ++index) {
    const u32 key = key_at(workspace.ordered_packet_indices[index]);
    min_key = std::min(min_key, key);
    max_key = std::max(max_key, key);
  }
  const u64 range = static_cast<u64>(max_key) - static_cast<u64>(min_key) + 1u;
  if (range > workspace.packet_partition_indices.size()) {
    return false;
  }

  const u32 bucket_count = static_cast<u32>(range);
  std::fill(workspace.packet_partition_indices.begin(),
            workspace.packet_partition_indices.begin() + bucket_count,
            0u);
  for (u32 index = 0u; index < packet_count; ++index) {
    const u32 bucket = key_at(workspace.ordered_packet_indices[index]) - min_key;
    workspace.packet_partition_indices[bucket] += 1u;
  }

  u32 prefix = 0u;
  for (u32 bucket = 0u; bucket < bucket_count; ++bucket) {
    const u32 count = workspace.packet_partition_indices[bucket];
    workspace.packet_partition_indices[bucket] = prefix;
    prefix += count;
  }
  for (u32 index = 0u; index < packet_count; ++index) {
    const u32 packet = workspace.ordered_packet_indices[index];
    const u32 bucket = key_at(packet) - min_key;
    workspace.ordered_packet_scratch[workspace.packet_partition_indices[bucket]++] = packet;
  }
  std::copy(workspace.ordered_packet_scratch.begin(),
            workspace.ordered_packet_scratch.begin() + packet_count,
            workspace.ordered_packet_indices.begin());
  return true;
}

template <typename KeyFn>
U64KeyRange MeasureU64KeyRange(const Workspace& workspace,
                               const u32 packet_count,
                               KeyFn key_at) {
  u64 min_key = key_at(workspace.ordered_packet_indices[0u]);
  u64 max_key = min_key;
  for (u32 index = 1u; index < packet_count; ++index) {
    const u64 key = key_at(workspace.ordered_packet_indices[index]);
    min_key = std::min(min_key, key);
    max_key = std::max(max_key, key);
  }
  return U64KeyRange{.min_key = min_key, .max_key = max_key};
}

u32 HighestDifferingByte(const U64KeyRange range) {
  u64 difference = range.min_key ^ range.max_key;
  u32 highest_byte = 0u;
  while (difference > 0xFFu) {
    difference >>= 8u;
    ++highest_byte;
  }
  return highest_byte;
}

template <typename KeyFn>
bool StableCountingOrderByMeasuredU64(Workspace& workspace,
                                      const u32 packet_count,
                                      const U64KeyRange key_range,
                                      KeyFn key_at) {
  const u64 key_width = key_range.max_key - key_range.min_key;
  if (key_width >= workspace.packet_partition_indices.size()) {
    return false;
  }

  const u32 bucket_count = static_cast<u32>(key_width + 1u);
  std::fill(workspace.packet_partition_indices.begin(),
            workspace.packet_partition_indices.begin() + bucket_count,
            0u);
  for (u32 index = 0u; index < packet_count; ++index) {
    const u32 bucket =
        static_cast<u32>(key_at(workspace.ordered_packet_indices[index]) - key_range.min_key);
    workspace.packet_partition_indices[bucket] += 1u;
  }

  u32 prefix = 0u;
  for (u32 bucket = 0u; bucket < bucket_count; ++bucket) {
    const u32 count = workspace.packet_partition_indices[bucket];
    workspace.packet_partition_indices[bucket] = prefix;
    prefix += count;
  }
  for (u32 index = 0u; index < packet_count; ++index) {
    const u32 packet = workspace.ordered_packet_indices[index];
    const u32 bucket = static_cast<u32>(key_at(packet) - key_range.min_key);
    workspace.ordered_packet_scratch[workspace.packet_partition_indices[bucket]++] = packet;
  }
  std::copy(workspace.ordered_packet_scratch.begin(),
            workspace.ordered_packet_scratch.begin() + packet_count,
            workspace.ordered_packet_indices.begin());
  return true;
}

template <typename KeyFn>
void StableRadixOrderByU32(Workspace& workspace,
                           const u32 packet_count,
                           KeyFn key_at) {
  std::array<u32, 256u> counts{};
  std::array<u32, 256u> offsets{};
  for (u32 byte = 0u; byte < 4u; ++byte) {
    counts.fill(0u);
    const u32 shift = byte * 8u;
    const std::vector<u32>& source =
        byte % 2u == 0u ? workspace.ordered_packet_indices : workspace.ordered_packet_scratch;
    std::vector<u32>& target =
        byte % 2u == 0u ? workspace.ordered_packet_scratch : workspace.ordered_packet_indices;
    for (u32 index = 0u; index < packet_count; ++index) {
      counts[(key_at(source[index]) >> shift) & 0xFFu] += 1u;
    }
    u32 prefix = 0u;
    for (u32 bucket = 0u; bucket < counts.size(); ++bucket) {
      offsets[bucket] = prefix;
      prefix += counts[bucket];
    }
    for (u32 index = 0u; index < packet_count; ++index) {
      const u32 packet = source[index];
      const u32 bucket = (key_at(packet) >> shift) & 0xFFu;
      target[offsets[bucket]++] = packet;
    }
  }
}

template <typename KeyFn>
void StableRadixOrderByU64(Workspace& workspace,
                           const u32 packet_count,
                           const U64KeyRange range,
                           KeyFn key_at) {
  std::array<u32, 256u> counts{};
  std::array<u32, 256u> offsets{};
  const u32 last_byte = HighestDifferingByte(range);
  for (u32 byte = 0u; byte <= last_byte; ++byte) {
    counts.fill(0u);
    const u32 shift = byte * 8u;
    const std::vector<u32>& source =
        byte % 2u == 0u ? workspace.ordered_packet_indices : workspace.ordered_packet_scratch;
    std::vector<u32>& target =
        byte % 2u == 0u ? workspace.ordered_packet_scratch : workspace.ordered_packet_indices;
    for (u32 index = 0u; index < packet_count; ++index) {
      counts[(key_at(source[index]) >> shift) & 0xFFu] += 1u;
    }
    u32 prefix = 0u;
    for (u32 bucket = 0u; bucket < counts.size(); ++bucket) {
      offsets[bucket] = prefix;
      prefix += counts[bucket];
    }
    for (u32 index = 0u; index < packet_count; ++index) {
      const u32 packet = source[index];
      const u32 bucket = (key_at(packet) >> shift) & 0xFFu;
      target[offsets[bucket]++] = packet;
    }
  }
  if (last_byte % 2u == 0u) {
    std::copy(workspace.ordered_packet_scratch.begin(),
              workspace.ordered_packet_scratch.begin() + packet_count,
              workspace.ordered_packet_indices.begin());
  }
}

template <typename KeyFn>
void StableOrderByU64(Workspace& workspace,
                      const u32 packet_count,
                      KeyFn key_at) {
  if (packet_count == 0u) {
    return;
  }
  const U64KeyRange range = MeasureU64KeyRange(workspace, packet_count, key_at);
  if (StableCountingOrderByMeasuredU64(workspace, packet_count, range, key_at)) {
    return;
  }
  StableRadixOrderByU64(workspace, packet_count, range, key_at);
}

} // namespace

void BuildWeightedPacketOrder(Workspace& workspace,
                              const ScheduleCompileRequest& request,
                              const std::span<const u64> resolved_work_units) {
  std::iota(workspace.ordered_packet_indices.begin(), workspace.ordered_packet_indices.end(), 0u);
  if (request.packet_count == 0u) {
    return;
  }
  const bool has_hints = HasPacketHints(request);
  if (has_hints) {
    const bool span_ordered = StableCountingOrderByU32(workspace, request.packet_count, [&](const u32 packet) {
      return std::numeric_limits<u32>::max() - request.packet_hints[packet].preferred_contiguous_span;
    });
    if (!span_ordered) {
      StableRadixOrderByU32(workspace, request.packet_count, [&](const u32 packet) {
        return std::numeric_limits<u32>::max() - request.packet_hints[packet].preferred_contiguous_span;
      });
    }
    const bool cache_ordered = StableCountingOrderByU32(workspace, request.packet_count, [&](const u32 packet) {
      return request.packet_hints[packet].cache_line_group;
    });
    if (!cache_ordered) {
      StableRadixOrderByU32(workspace, request.packet_count, [&](const u32 packet) {
        return request.packet_hints[packet].cache_line_group;
      });
    }
  }
  StableOrderByU64(workspace, request.packet_count, [&](const u32 packet) {
    return std::numeric_limits<u64>::max() - resolved_work_units[packet];
  });
  if (has_hints) {
    const bool locality_ordered = StableCountingOrderByU32(workspace, request.packet_count, [&](const u32 packet) {
      return request.packet_hints[packet].locality_bucket_id;
    });
    if (!locality_ordered) {
      StableRadixOrderByU32(workspace, request.packet_count, [&](const u32 packet) {
        return request.packet_hints[packet].locality_bucket_id;
      });
    }
  }
}

} // namespace rund::kernel::workspace_detail
