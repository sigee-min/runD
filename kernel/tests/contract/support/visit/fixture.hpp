#pragma once

#include "../visits.hpp"

namespace kernel_contract_test {

struct VisitFixture {
  std::array<std::atomic<rund::kernel::u32>, kMaxVisitedPackets> visits{};
  std::array<rund::kernel::u32, kMaxVisitedPackets> observed_packets{};
  std::atomic<rund::kernel::u32> observed_count{0u};
  VisitContext context{
      .visits = &visits,
      .observed_packets = &observed_packets,
      .observed_count = &observed_count,
      .packet_count = 0u,
  };

  void Reset(const rund::kernel::u32 packet_count) {
    ResetVisits(visits);
    observed_count.store(0u, std::memory_order_relaxed);
    context.packet_count = packet_count;
  }

  bool VisitedOnce(const rund::kernel::u32 packet_count) const {
    return EachPacketVisitedOnce(visits, packet_count);
  }
};

} // namespace kernel_contract_test
