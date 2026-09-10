#pragma once

#include <rund/compute/abi/state.hpp>
#include <rund/compute/fixed.hpp>
#include <rund/compute/pipeline/memory.hpp>
#include <rund/compute/status.hpp>
#include <rund/compute/telemetry.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace rund::compute {
class VirtualBacking;

enum class PageOrigin : std::uint8_t { Begin, End };

struct GraphPageMapEntry final {
  std::uint32_t input{};
  std::uint32_t target_local{};
  std::uint32_t source_local{};
  PageOrigin origin{PageOrigin::Begin};

  [[nodiscard]] constexpr bool
  operator==(const GraphPageMapEntry &) const noexcept = default;
};

// Optional graph-page binding authored at the public graph boundary. The
// entries are copied into the prepared residency plan; no caller span is
// retained by execution or Authority.
struct GraphPageMap final {
  std::uint64_t graph_hi{};
  std::uint64_t graph_lo{};
  std::span<const GraphPageMapEntry> entries{};

  [[nodiscard]] constexpr bool empty() const noexcept {
    return entries.empty();
  }
};

// Byte budgets are policy; frame count is a derived execution fact. Zero asks
// the Device residency authority for its bounded default working set.
struct ResidencyConfig final {
  std::uint64_t device_resident_bytes{};
  // Capacity of the VSM Host-resident working set. On CPU this admits the
  // executable Host frame banks directly. On accelerators it admits bounded
  // Host supply/writeback frames; it is never an unbounded backing promise.
  std::uint64_t host_resident_bytes{};
};

namespace detail {

[[nodiscard]] Result<std::shared_ptr<VirtualBufferState>>
make_virtual_buffer(std::uint64_t count, std::uint64_t element_bytes, Type type,
                    FixedFormat format,
                    std::shared_ptr<VirtualBacking> backing) noexcept;
[[nodiscard]] Result<std::shared_ptr<VirtualBacking>>
make_resident_virtual_backing(const std::shared_ptr<DeviceState> &device,
                              Type type, std::uint64_t count) noexcept;
[[nodiscard]] bool
valid_virtual_buffer(const std::shared_ptr<VirtualBufferState> &state) noexcept;
[[nodiscard]] std::uint64_t
virtual_buffer_size(const std::shared_ptr<VirtualBufferState> &state) noexcept;

[[nodiscard]] Result<std::shared_ptr<VirtualPipelineState>>
prepare_virtual_pipeline(
    const std::shared_ptr<ProgramState> &program,
    std::span<const std::shared_ptr<VirtualBufferState>> inputs,
    const std::shared_ptr<VirtualBufferState> &output,
    ResidencyConfig config) noexcept;
[[nodiscard]] Result<std::shared_ptr<VirtualPipelineState>>
prepare_virtual_pipeline(
    const std::shared_ptr<ProgramState> &program,
    std::span<const std::shared_ptr<VirtualBufferState>> inputs,
    const std::shared_ptr<VirtualBufferState> &output, ResidencyConfig config,
    GraphPageMap page_map) noexcept;
[[nodiscard]] inline Result<std::shared_ptr<VirtualPipelineState>>
prepare_virtual_pipeline(const std::shared_ptr<ProgramState> &program,
                         const std::shared_ptr<VirtualBufferState> &input,
                         const std::shared_ptr<VirtualBufferState> &output,
                         const ResidencyConfig config) noexcept {
  const std::array inputs{input};
  return prepare_virtual_pipeline(program, inputs, output, config);
}
[[nodiscard]] bool valid_virtual_pipeline(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept;
[[nodiscard]] Status run_virtual_pipeline(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept;
[[nodiscard]] Status
run_virtual_pipeline(const std::shared_ptr<VirtualPipelineState> &state,
                     std::uint64_t active_count) noexcept;
[[nodiscard]] Status begin_virtual_pipeline_samples(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept;
[[nodiscard]] Status end_virtual_pipeline_samples(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept;
[[nodiscard]] Stats virtual_pipeline_stats(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept;
[[nodiscard]] MemoryStats virtual_pipeline_memory(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept;
[[nodiscard]] PipelinePlan virtual_pipeline_plan(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept;
[[nodiscard]] Result<telemetry::Profile> virtual_pipeline_profile(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept;

} // namespace detail
} // namespace rund::compute
