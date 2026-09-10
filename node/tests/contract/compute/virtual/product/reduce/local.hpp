#pragma once

#include "../backing.hpp"
#include "../route.hpp"

#include "../../../../target/selection.hpp"
#include "../../../allocation.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include "src/compute/flow/state.hpp"
#include "src/compute/graph/compile/slice.hpp"
#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/state.hpp"
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include "src/accel/vulkan/kernel/pipeline/residency/local.hpp"
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace rund_node_test_virtual::product::reduce {

inline constexpr std::size_t ReduceFrameElements = 16u;
inline constexpr std::size_t ReduceElements = 137u;
inline constexpr std::size_t ReducePages =
    (ReduceElements + ReduceFrameElements - 1u) / ReduceFrameElements;
inline constexpr std::size_t GraphWarmRuns = 60u;
inline constexpr std::uint64_t ActiveMapped = 71u;

using U64Program = rund::compute::Program<std::uint64_t(std::uint64_t)>;
using U64Pipeline =
    rund::compute::VirtualPipeline<std::uint64_t(std::uint64_t)>;
using U64Buffer = rund::compute::VirtualBuffer<std::uint64_t>;

struct ReduceFixture final {
  std::optional<rund::compute::Device> device{};
  std::optional<U64Program> sum_program{};
  std::optional<U64Program> mapped_reduce_program{};
  std::optional<U64Pipeline> mapped_reduce{};
  std::shared_ptr<MemoryVirtualBacking> input_backing{};
  std::shared_ptr<MemoryVirtualBacking> output_backing{};
  std::optional<U64Buffer> input{};
  std::optional<U64Buffer> output{};
  std::array<std::uint64_t, ReduceElements> values{};
  std::uint64_t expected_sum{};
  bool backend_unavailable{};
};

[[nodiscard]] int InitializeReduceFixture(ReduceFixture &fixture,
                                           rund::compute::Backend backend);

[[nodiscard]] bool same_fixed_memory(const rund::compute::MemoryStats &left,
                                     const rund::compute::MemoryStats &right)
    noexcept;

[[nodiscard]] bool allocation_boundary_exact(
    rund::compute::Backend backend,
    std::uint64_t observed_allocations) noexcept;

[[nodiscard]] bool observe_u64(MemoryVirtualBacking &backing,
                               std::uint64_t &value) noexcept;
[[nodiscard]] bool observe_i32(MemoryVirtualBacking &backing,
                               std::int32_t &value) noexcept;

class DelayedGraphBacking final : public rund::compute::VirtualBacking {
public:
  explicit DelayedGraphBacking(std::span<const std::byte> bytes);

  [[nodiscard]] std::uint64_t size_bytes() const noexcept override;
  [[nodiscard]] rund::compute::VirtualBackingTier
  tier() const noexcept override;
  [[nodiscard]] std::uint32_t max_parallel_reads() const noexcept override;
  [[nodiscard]] rund::compute::Status
  read(std::uint64_t offset, std::span<std::byte> output) noexcept override;
  [[nodiscard]] rund::compute::Status
  write(std::uint64_t, std::span<const std::byte>) noexcept override;
  [[nodiscard]] std::uint32_t max_active_reads() const noexcept;

private:
  std::vector<std::byte> bytes_;
  std::atomic<std::uint32_t> active_reads_{};
  std::atomic<std::uint32_t> max_active_reads_{};
};

[[nodiscard]] int RunBasicReduce(ReduceFixture &fixture,
                                  rund::compute::Backend backend);
[[nodiscard]] int RunSequenceReduce(ReduceFixture &fixture,
                                     rund::compute::Backend backend);
[[nodiscard]] int RunLendingReduce(ReduceFixture &fixture,
                                   rund::compute::Backend backend);
[[nodiscard]] int RunDirectReduce(ReduceFixture &fixture,
                                  rund::compute::Backend backend);

} // namespace rund_node_test_virtual::product::reduce
