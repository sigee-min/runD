#pragma once

#include "model.hpp"

#include <cstdint>

namespace rund_node_test_persistent_product {

struct RunSample final {
  rund::compute::Backend backend{rund::compute::Backend::Unavailable};
  std::uint64_t coordinates{};
  rund::compute::Status status{
      rund::compute::Status::fail(rund::compute::Reason::BackendUnsupported)};
  ProductRouteObservation observation{};
  const PreparedProduct *product{};
  std::uint64_t queue_before{};
  std::uint64_t queue_after{};
  std::uint64_t command_submits{};
  bool queue_read{};
  PublicationSnapshot before_primary{};
  PublicationSnapshot after_primary{};
  PublicationSnapshot before_alternate{};
  PublicationSnapshot after_alternate{};
  std::uint64_t version_before{};
  std::uint64_t version_after{};
  std::uint64_t recovery_after{};
  rund::compute::ResidencyStats stats{};
  std::uint64_t input_read_calls{};
  std::uint64_t input_read_ranges{};
  std::uint64_t pages{};
  std::uint64_t backing_input_bytes{};
  std::uint64_t materialized_input_bytes{};
};

[[nodiscard]] inline std::uint64_t
PhysicalSubmits(const std::uint64_t coordinates) noexcept {
  // The product fixture opts into BackendChunked through Persistent input
  // parallel-read capability.  Keep this separate from the lower-level
  // OneSubmit contract, whose native count is exactly one.
  return coordinates == 0u
             ? 0u
             : coordinates / 2u + static_cast<std::uint64_t>(coordinates % 2u != 0u);
}

[[nodiscard]] bool ExactRun(const RunSample &) noexcept;

[[nodiscard]] bool ExactPublish(PublicationSnapshot, PublicationSnapshot,
                                PublicationSnapshot, PublicationSnapshot,
                                std::uint64_t, std::uint64_t,
                                std::uint64_t) noexcept;

[[nodiscard]] bool ExactMode(
    const ProductRouteObservation &,
    rund::node::accel::detail::PersistentResidencySlidingMode) noexcept;

[[nodiscard]] bool ExactNativeFinal(const ProductRouteObservation &,
                                    std::uint64_t) noexcept;
[[nodiscard]] bool ExactAuthorityFinal(const ProductRouteObservation &,
                                       std::uint64_t) noexcept;
[[nodiscard]] bool ExactAuthorityIo(const ProductRouteObservation &,
                                    std::uint64_t, std::uint64_t, std::uint64_t,
                                    std::uint64_t) noexcept;

} // namespace rund_node_test_persistent_product
