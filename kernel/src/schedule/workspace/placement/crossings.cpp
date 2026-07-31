#include "local.hpp"

namespace rund::kernel::workspace_placement {
namespace {

using BucketSelector = u32 (*)(const PacketPlacementHint& hint);

u32 LocalityBucket(const PacketPlacementHint& hint) {
  return hint.locality_bucket_id;
}

u32 CountHintBucketCrossings(const ScheduleCompileRequest& request,
                             const std::span<const u32> ordered_packets,
                             const BucketSelector bucket_of) {
  if (request.packet_hints.size() != static_cast<std::size_t>(request.packet_count)) {
    return 0u;
  }
  if (ordered_packets.empty()) {
    u32 crossings = 0u;
    for (u32 packet = 1u; packet < request.packet_count; ++packet) {
      if (bucket_of(request.packet_hints[packet - 1u]) !=
          bucket_of(request.packet_hints[packet])) {
        crossings += 1u;
      }
    }
    return crossings;
  }
  u32 crossings = 0u;
  for (std::size_t index = 1u; index < ordered_packets.size(); ++index) {
    const u32 prev = ordered_packets[index - 1u];
    const u32 next = ordered_packets[index];
    if (bucket_of(request.packet_hints[prev]) != bucket_of(request.packet_hints[next])) {
      crossings += 1u;
    }
  }
  return crossings;
}

} // namespace

u32 CountLocalityBucketCrossings(const ScheduleCompileRequest& request,
                                 const std::span<const u32> ordered_packets) {
  return CountHintBucketCrossings(request, ordered_packets, LocalityBucket);
}

} // namespace rund::kernel::workspace_placement
