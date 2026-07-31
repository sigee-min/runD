#pragma once

#include <kernel/core/model.hpp>

#include <array>
#include <atomic>

namespace kernel_contract_test {

constexpr rund::kernel::u32 kMaxVisitedPackets = 32u;

struct VisitContext {
  std::array<std::atomic<rund::kernel::u32>, kMaxVisitedPackets>* visits = nullptr;
  std::array<rund::kernel::u32, kMaxVisitedPackets>* observed_packets = nullptr;
  std::atomic<rund::kernel::u32>* observed_count = nullptr;
  rund::kernel::u32 packet_count = 0u;
};

inline void ResetVisits(std::array<std::atomic<rund::kernel::u32>, kMaxVisitedPackets>& visits) {
  for (std::atomic<rund::kernel::u32>& visit : visits) {
    visit.store(0u, std::memory_order_relaxed);
  }
}

inline bool EachPacketVisitedOnce(const std::array<std::atomic<rund::kernel::u32>, kMaxVisitedPackets>& visits,
                                  const rund::kernel::u32 packet_count) {
  for (rund::kernel::u32 packet = 0u; packet < packet_count; ++packet) {
    if (visits[packet].load(std::memory_order_relaxed) != 1u) {
      return false;
    }
  }
  return true;
}

inline void MarkVisitedPackets(void* raw_context, const rund::kernel::Partition& partition) {
  auto* const context = static_cast<VisitContext*>(raw_context);
  if (context == nullptr || context->visits == nullptr) {
    return;
  }
  for (rund::kernel::u32 offset = 0u; offset < partition.packet_count(); ++offset) {
    const rund::kernel::u32 packet = partition.packet_at(offset);
    if (packet >= context->packet_count) {
      continue;
    }
    (*context->visits)[packet].fetch_add(1u, std::memory_order_relaxed);
    if (context->observed_packets != nullptr && context->observed_count != nullptr) {
      const rund::kernel::u32 write_index =
          context->observed_count->fetch_add(1u, std::memory_order_relaxed);
      if (write_index < kMaxVisitedPackets) {
        (*context->observed_packets)[write_index] = packet;
      }
    }
  }
}

} // namespace kernel_contract_test
