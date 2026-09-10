#pragma once

#include "proof.hpp"

#include <accel/check.hpp>

#include <cstdint>
#include <memory>
#include <span>

namespace rund::node::accel::detail {

struct MapRecurrence;
struct BackendBatchEntry;

struct ServiceFreeDirectProjection final {
  rund::AccelCheck check{false, "accel_kernel_pipeline_invalid"};
  std::shared_ptr<const ServiceFreeDirectProof> proof{};
};

[[nodiscard]] ServiceFreeDirectProjection
ProjectServiceFreeDirectProof(const MapRecurrence &recurrence,
                              std::span<const BackendBatchEntry> entries,
                              std::shared_ptr<const void> semantic_owner,
                              std::uint64_t pipeline_fingerprint_hi,
                              std::uint64_t pipeline_fingerprint_lo) noexcept;

} // namespace rund::node::accel::detail
