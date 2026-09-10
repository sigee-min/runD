#pragma once

#include "src/compute/virtual/run/device_vsm/route/proof.hpp"

#include <limits>
#include <type_traits>
#include <variant>

namespace rund_node_test_virtual::product::route_detail {

[[nodiscard]] inline bool proof_tamper_rejected(
    const rund::compute::detail::VirtualDeviceVsmRouteProof &proof) noexcept {
  using namespace rund::compute::detail;
  if (!proof.valid())
    return false;
  // Mutate the actual selected payload, not a parallel tag or shape mirror.
  auto route = proof.route();
  std::visit(
      [](auto &value) noexcept {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, VirtualDeviceVsmWindowRing>)
          value.window.frame_capacity = 0u;
        else
          value.frame_capacity = 0u;
      },
      route);
  if (VirtualDeviceVsmRouteProof{route, proof.stamp_hi, proof.stamp_lo}.valid())
    return false;
  if (proof.kind() == VirtualDeviceVsmRouteKind::Direct)
    return proof.endpoint() == VirtualDeviceVsmEndpoint::Invalid &&
           proof.window() == nullptr;
  if (proof.kind() == VirtualDeviceVsmRouteKind::StagedLoop)
    return proof.endpoint() == VirtualDeviceVsmEndpoint::Staged &&
           proof.window() == nullptr;
  route = proof.route();
  std::visit(
      [](auto &value) noexcept {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, VirtualDeviceVsmWindowRing>)
          value.window.endpoint = VirtualWindowPreflightEndpoint::Invalid;
        else if constexpr (std::is_same_v<T, VirtualDeviceVsmGraphResident>)
          value.endpoint = VirtualDeviceVsmEndpoint::Invalid;
      },
      route);
  return !VirtualDeviceVsmRouteProof{route, proof.stamp_hi, proof.stamp_lo}
              .valid();
}

[[nodiscard]] inline bool check_route_proof() noexcept {
  using namespace rund::compute::detail;
  const VirtualWindowPreflight window{
      .mode = VirtualWindowPreflightMode::WindowRing,
      .endpoint = VirtualWindowPreflightEndpoint::Resident,
      .page_count = 3u,
      .frame_capacity = 2u,
      .input_bytes = 48u,
      .output_bytes = 48u,
      .input_frame_bytes = 16u,
      .output_frame_bytes = 16u,
      .frame_bytes = 32u,
      .device_storage_bytes = 128u,
      .host = {.input = 3u, .output = 2u, .storage_bytes = 160u},
  };
  const VirtualDeviceVsmRouteProof proofs[]{
      VirtualDeviceVsmRouteProof{VirtualDeviceVsmDirect{3u, 2u}, 1u},
      VirtualDeviceVsmRouteProof{VirtualDeviceVsmStagedLoop{3u, 2u}, 1u},
      VirtualDeviceVsmRouteProof{VirtualDeviceVsmWindowRing{window}, 1u},
      VirtualDeviceVsmRouteProof{
          VirtualDeviceVsmGraphResident{3u, 2u,
                                        VirtualDeviceVsmEndpoint::Resident},
          1u},
      VirtualDeviceVsmRouteProof{VirtualDeviceVsmGraphResident{
                                     3u, 2u, VirtualDeviceVsmEndpoint::Staged},
                                 1u},
  };
  for (const auto &proof : proofs) {
    if (proof.page_count() != 3u || proof.frame_capacity() != 2u ||
        !proof_tamper_rejected(proof))
      return false;
    auto unsigned_proof = proof;
    unsigned_proof.stamp_hi = unsigned_proof.stamp_lo = 0u;
    if (unsigned_proof.valid())
      return false;
  }
  for (auto input : {0ull, 1ull, 0x100000000ull}) {
    if (VirtualDeviceVsmRouteProof{VirtualDeviceVsmDirect{input, 2u}, 1u}
            .valid())
      return false;
  }
  auto broken = window;
  broken.host.input = 2u;
  if (VirtualDeviceVsmRouteProof{VirtualDeviceVsmWindowRing{broken}, 1u}
          .valid())
    return false;
  broken = window;
  broken.input_frame_bytes = std::numeric_limits<std::uint64_t>::max();
  if (VirtualDeviceVsmRouteProof{VirtualDeviceVsmWindowRing{broken}, 1u}
          .valid())
    return false;
  broken = window;
  broken.device_storage_bytes -= 1u;
  return !VirtualDeviceVsmRouteProof{}.valid() &&
         !VirtualDeviceVsmRouteProof{VirtualDeviceVsmWindowRing{broken}, 1u}
              .valid();
}

} // namespace rund_node_test_virtual::product::route_detail
