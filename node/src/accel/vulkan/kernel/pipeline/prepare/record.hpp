#pragma once

#include "../state.hpp"

#include "../../../command/capture.hpp"
#include "../../../command/timestamp.hpp"

#include "../../../../kernel/backend/pipeline/failure.hpp"
#include "../../../../kernel/recurrence.hpp"

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanPipelineRecordEntry final {
  const BackendRun *run{};
  std::shared_ptr<void> prepared;
  std::optional<BackendWindow> window;
  std::uint32_t transducer{NoTileTransducer};
  std::uint32_t template_index{};
  std::uint32_t occurrence_index{};
  std::uint32_t step_index{};
};

struct VulkanPipelineRecordRecipe final {
  std::vector<VulkanPipelineRecordEntry> entries;
  std::vector<std::uint8_t> barriers;
  std::vector<std::uint8_t> transducer_ready;
  std::vector<VulkanPipelineCanonicalStatus> canonical;
  std::vector<PreparedProgramStatusSlice> status_steps;
  std::vector<PreparedProgramStatusSlice> telemetry_steps;
  std::array<PreparedProgramStatusSlice, PreparedPipelineStepCapacity>
      status_ranges{};
  std::array<PreparedProgramStatusSlice, PreparedPipelineStepCapacity>
      telemetry_ranges{};
  PreparedPipelineStatusLayout status{};
  std::size_t template_count{};
  std::uint64_t window_dispatches{};
  std::uint64_t window_gate_count{};
  bool recurrence{};
};

struct VulkanPipelineRecordSlice final {
  std::size_t first{};
  std::size_t count{};
  bool open{};
  bool close{};
  VulkanDispatchCapture *capture{};
  std::uint32_t capture_owner{};
};

[[nodiscard]] rund::AccelCheck
DescribeVulkanRouteDispatches(const VulkanKernelResources &,
                              std::uint64_t &total,
                              std::uint64_t &indirect) noexcept;

[[nodiscard]] rund::AccelCheck MakeVulkanPipelineRecordRecipe(
    std::span<const BackendBatchEntry> entries,
    std::span<const std::uint8_t> barriers,
    std::span<const TileTransducer> transducers,
    std::span<const VulkanPipelineCanonicalStatus> canonical,
    std::span<const PreparedProgramStatusSlice> status_steps,
    std::span<const PreparedProgramStatusSlice> telemetry_steps,
    const std::array<PreparedProgramStatusSlice, PreparedPipelineStepCapacity>
        &status_ranges,
    const std::array<PreparedProgramStatusSlice, PreparedPipelineStepCapacity>
        &telemetry_ranges,
    const PreparedPipelineStatusLayout &status, std::size_t template_count,
    std::uint64_t window_dispatches, std::uint64_t window_gate_count,
    bool recurrence, VulkanPipelineRecordRecipe &recipe) noexcept;

[[nodiscard]] rund::AccelCheck
RecordVulkanPipeline(VulkanPipeline &pipeline, VulkanCommand &command,
                     CommandKind kind, bool replay,
                     PreparedPipelineFailureContext *failure = nullptr,
                     VulkanTimestampCapture *timestamps = nullptr,
                     const VulkanPipelineRecordSlice *slice = nullptr) noexcept;

[[nodiscard]] std::uint64_t VulkanPipelineRecordHostBytes(
    const VulkanPipelineRecordRecipe &recipe) noexcept;

[[nodiscard]] bool VulkanPipelineRecordHostBytes(
    const PreparedKernelPipelineReservation &reservation,
    std::uint64_t &bytes) noexcept;

#endif

} // namespace rund::node::accel::detail
