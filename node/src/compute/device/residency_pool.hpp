#pragma once

#include "residency.hpp"
#include "residency_prefetch.hpp"

#include <rund/compute/abi/model.hpp>
#include <rund/storage.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace rund::compute::detail {
struct BufferState;
struct DeviceState;

namespace residency {

struct PoolLayout final {
  Type input_type{Type::I32};
  FixedFormat input_format{};
  Type output_type{Type::I32};
  FixedFormat output_format{};
  std::uint64_t input_page_bytes{};
  std::uint64_t output_page_bytes{};
  std::uint32_t frame_capacity{};

  [[nodiscard]] constexpr bool
  operator==(const PoolLayout &) const noexcept = default;
};

class Pool final {
public:
  ~Pool();
  PoolLayout layout{};
  std::shared_ptr<DeviceState> device;
  // Canonical device-tier frames survive epoch execution and are the sole
  // eviction/writeback source. Execution arenas are disposable banks bound
  // into the prepared Pipeline and may be overwritten by every submission.
  std::shared_ptr<BufferState> cache_input;
  std::shared_ptr<BufferState> cache_output;
  std::shared_ptr<BufferState> input;
  std::shared_ptr<BufferState> output;
  std::unique_ptr<std::byte[]> staging;
  std::uint64_t staging_bytes{};
  Authority authority;
  // Two fixed workers let a persistent backing keep e+1 and e+2 in flight.
  // Host backings use only one lane; no run allocates queue or payload state.
  std::array<Prefetcher, 2u> prefetch;
  // One Pool has one mutable mapping table and one prefetch buffer. Holding
  // this lease for the complete run prevents cross-Pipeline epoch interleave.
  std::mutex execution_gate;
  // The global pool is charged to the Device's sole Pipeline budget for its
  // complete lifetime; Pipeline-local plans only retain a shared reference.
  storage::Reservation memory;
  std::uint64_t host_bytes{};
  bool host_accounted{};
};

class Registry final {
public:
  [[nodiscard]] std::shared_ptr<Pool>
  acquire(const std::shared_ptr<DeviceState> &device,
          const PoolLayout &layout) noexcept;

private:
  std::mutex gate_;
  std::vector<std::weak_ptr<Pool>> pools_;
};

} // namespace residency
} // namespace rund::compute::detail
