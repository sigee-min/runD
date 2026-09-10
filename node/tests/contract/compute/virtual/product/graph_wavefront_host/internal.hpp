#pragma once

#include "../backing.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace rund_node_test_virtual::product {
struct ProductRouteObservation;
}

namespace rund_node_test_virtual::product::graph_wavefront_host {

inline constexpr std::size_t FrameElements = 16u;
inline constexpr std::size_t ElementCount = 73u;
inline constexpr std::size_t GraphPageCount =
    (ElementCount + FrameElements - 1u) / FrameElements;
inline constexpr std::size_t StageCount = 4u;
inline constexpr std::size_t InputCount = 3u;
inline constexpr std::size_t PageBytes = FrameElements * sizeof(std::uint64_t);
// Each authored stage remains below the kernel expression limit, while the
// public Map DAG composed across all four stages exceeds it. This keeps the
// prepared route as the ordinary tiled GraphPointwise path without changing
// the sealed four-stage topology.
inline constexpr std::size_t StageLeafCount = 340u;

using Program = rund::compute::Program<std::uint64_t(
    std::uint64_t, std::uint64_t, std::uint64_t)>;
using Pipeline = rund::compute::VirtualPipeline<std::uint64_t(
    std::uint64_t, std::uint64_t, std::uint64_t)>;

struct PairReadGate final {
  mutable std::mutex gate;
  std::condition_variable ready;
  std::size_t page_bytes{};
  bool slow_blocked{};
  bool slow_released{};
  bool fast_started{};
  bool fast_completed{};
  bool slow_completed{};
  bool reversed{};
  bool premature_reuse{};
  bool timed_out{};
};

class ReversedGraphBacking final : public rund::compute::VirtualBacking {
public:
  ReversedGraphBacking(std::size_t logical_bytes, std::size_t page_bytes,
                       std::shared_ptr<PairReadGate> gate, bool slow);

  [[nodiscard]] std::uint64_t size_bytes() const noexcept override;
  [[nodiscard]] rund::compute::Status
  read(std::uint64_t offset, std::span<std::byte> output) noexcept override;
  [[nodiscard]] rund::compute::Status
  write(std::uint64_t offset,
        std::span<const std::byte> input) noexcept override;

  [[nodiscard]] bool seed(std::span<const std::byte> input) noexcept;
  [[nodiscard]] BackingFacts facts() const noexcept;
  [[nodiscard]] bool reversed() const noexcept;
  [[nodiscard]] bool premature_reuse() const noexcept;
  [[nodiscard]] bool both_callbacks_started_before_release() const noexcept;

private:
  [[nodiscard]] bool contains(std::uint64_t offset,
                              std::size_t bytes) const noexcept;

  std::vector<std::byte> bytes_;
  std::size_t logical_bytes_{};
  std::size_t page_bytes_{};
  std::shared_ptr<PairReadGate> gate_;
  bool slow_{};
  bool blocked_once_{};
  mutable std::mutex data_gate_;
  BackingFacts facts_{};
};

struct Case final {
  std::array<std::uint64_t, ElementCount> first_values{};
  std::array<std::uint64_t, ElementCount> second_values{};
  std::array<std::uint64_t, ElementCount> third_values{};
  std::array<std::uint64_t, ElementCount> expected{};
  std::shared_ptr<MemoryVirtualBacking> first_backing;
  std::shared_ptr<ReversedGraphBacking> second_backing;
  std::shared_ptr<ReversedGraphBacking> third_backing;
  std::shared_ptr<MemoryVirtualBacking> output_backing;
  Pipeline pipeline;
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state;
};

struct Preparation final {
  std::unique_ptr<Case> value{};
  int reason{};
};

[[nodiscard]] rund::compute::Result<Program>
build_program(const rund::compute::Device &);
[[nodiscard]] std::uint64_t stage_offset(std::uint64_t first) noexcept;
[[nodiscard]] bool validate_program(const Program &) noexcept;
[[nodiscard]] Preparation prepare_case(const rund::compute::Device &);

[[nodiscard]] bool check_graph_admission(
    const rund::compute::detail::VirtualPipelineState &) noexcept;
[[nodiscard]] bool
check_graph_replay(const rund::compute::detail::VirtualPipelineState &,
                   std::uint64_t capacity, std::uint64_t batches) noexcept;
[[nodiscard]] bool validate_case(
    Case &, rund::compute::Backend, const rund::compute::Status &,
    std::uint64_t initial_version,
    const rund_node_test_virtual::product::ProductRouteObservation &) noexcept;

} // namespace rund_node_test_virtual::product::graph_wavefront_host
