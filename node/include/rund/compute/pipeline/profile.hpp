#pragma once

#include <rund/compute/graph/info.hpp>
#include <rund/compute/pipeline/coordinate.hpp>
#include <rund/compute/stats.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace rund::compute {

enum class PipelineProfile : std::uint8_t {
  None,
  Steps,
};

enum class StepClock : std::uint8_t {
  Unavailable,
  HostSteady,
  Device,
};

enum class StepTimingRelation : std::uint8_t {
  Unavailable,
  Exclusive,
  NonAdditive,
};

struct StepTiming final {
  std::uint64_t duration_ns{};
  std::uint64_t sample_count{};
  StepClock clock{StepClock::Unavailable};
  StepTimingRelation relation{StepTimingRelation::Unavailable};

  [[nodiscard]] constexpr bool available() const noexcept {
    return clock != StepClock::Unavailable &&
           relation != StepTimingRelation::Unavailable &&
           sample_count != 0u;
  }

  [[nodiscard]] constexpr bool saturated() const noexcept {
    constexpr std::uint64_t limit = std::numeric_limits<std::uint64_t>::max();
    return available() &&
           (duration_ns == limit || sample_count == limit);
  }
};

struct PipelineStepStats final {
  std::uint64_t sample_count{};
  std::uint64_t original_dispatches{};
  std::uint64_t final_dispatches{};
  std::uint64_t barrier_count{};
  std::uint32_t worker_count{};
  std::uint32_t participating_workers{};
  std::uint64_t tile_count{};
  std::uint64_t tile_size{};
  std::uint64_t vector_chunks{};
  std::uint64_t tail_chunks{};
  std::uint64_t workgroup_count{};
  std::uint64_t work_item_count{};
  ControlStats control{};

  [[nodiscard]] constexpr bool available() const noexcept {
    return sample_count != 0u;
  }
};

struct PipelineStepProfile final {
  static constexpr std::uint32_t no_coordinate =
      std::numeric_limits<std::uint32_t>::max();

  std::uint32_t index{};
  std::uint32_t iteration{};
  std::uint32_t iteration_bound{1u};
  std::uint32_t outer_window{no_coordinate};
  std::uint32_t outer_window_bound{};
  std::uint32_t inner_iteration{no_coordinate};
  std::uint32_t inner_iteration_bound{};
  PipelineNestedPhase nested_phase{PipelineNestedPhase::None};
  graph::Fingerprint program{};
  StepTiming timing{};
  PipelineStepStats execution{};
  MemoryStats memory{};
};

struct PipelineProfileSnapshot final {
  Stats execution{};
  MemoryStats memory{};
  MemoryStats shared_memory{};
  StepTiming observation{};
  std::uint64_t referenced_resource_bytes{};
  std::uint64_t instrumentation_command_count{};
  std::uint64_t instrumentation_byte_count{};
  std::size_t written{};
  std::size_t total{};

  [[nodiscard]] constexpr bool truncated() const noexcept {
    return written < total;
  }
};

static_assert(std::is_trivially_copyable_v<StepTiming>);
static_assert(std::is_trivially_copyable_v<PipelineStepStats>);
static_assert(std::is_trivially_copyable_v<PipelineStepProfile>);
static_assert(std::is_trivially_copyable_v<PipelineProfileSnapshot>);

} // namespace rund::compute
