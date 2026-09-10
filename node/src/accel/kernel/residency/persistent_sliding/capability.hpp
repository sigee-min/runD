#pragma once

#include "../sliding.hpp"

#include <accel/check.hpp>

#include <cstdint>

namespace rund::node::accel::detail {

inline constexpr std::uint8_t PersistentResidencySlidingCapacity =
    static_cast<std::uint8_t>(ResidencySlidingCapacity);

enum class PersistentResidencySlidingMode : std::uint8_t {
  OneSubmit,
  BackendChunked,
};

// Product-only capability. `whole_run_preencoded` admits the current O(Q)
// intermediate lowering.  It is deliberately distinct from
// the stronger device-generated, fixed-native-storage recurrence required for
// a genuinely GPU-driven product path.
struct PersistentResidencySlidingCapability final {
  rund::AccelCheck check{false, "accel_kernel_pipeline_invalid"};
  ResidencySlidingMemory memory{ResidencySlidingMemory::HostCoherent};
  std::uint64_t retained_bytes{};
  std::uint64_t transient_bytes{};
  std::uint8_t width{};
  // Width is the role/bank width. Chunk span is an independent service limit.
  std::uint8_t max_chunk_span{2u};
  bool whole_run_preencoded{};
  bool device_generated_recurrence{};
  bool fixed_native_storage{};
  // Backend-native command storage is only half of the fixed-W proof. Common
  // authored occurrence/control rows must also be independent of Q.
  bool fixed_common_storage{};
  bool host_epoch_callbacks_zero{};
  // This is the sole selected submit-shape authority: OneSubmit is one native
  // submit, while BackendChunked is ceil(Q/2) native submits.
  PersistentResidencySlidingMode mode{
      PersistentResidencySlidingMode::OneSubmit};
};

} // namespace rund::node::accel::detail
