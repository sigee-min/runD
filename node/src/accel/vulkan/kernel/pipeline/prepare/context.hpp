#pragma once

#include "../evidence.hpp"
#include "../recurrence.hpp"
#include "../../../../kernel/backend/pipeline/failure.hpp"
#include "../../../../kernel/prepared/template/registry.hpp"
#include "../../../../kernel/recurrence.hpp"
#include "capacity.hpp"
#include "record.hpp"

#include <array>
#include <memory>
#include <span>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

// Shared state for the cold Pipeline transaction.  The parent prepare TU
// routes this state through semantic leaves; no leaf owns a second copy of the
// route, status, telemetry, publication, or residency policy.
struct VulkanPipelinePreparation final {
  std::span<const BackendBatchEntry> templates{};
  std::span<const BackendBatchEntry> entries{};
  std::span<const std::uint8_t> barriers{};
  std::span<const TileTransducer> transducers{};
  std::span<const BackendPublish> publications{};
  PreparedKernelTemplateRegistry *registry{};
  PreparedPipelineStatusLayout *status{};
  bool profile_steps{};
  PreparedPipelineMemoryMeter *memory_meter{};
  std::shared_ptr<void> *prepared{};
  PreparedPipelineMemory *memory{};
  PreparedPipelineFailureContext *failure_context{};

  MapRecurrence recurrence{};
  std::shared_ptr<VulkanPipeline> pipeline{};
  std::vector<VulkanPipelineCanonicalStatus> canonical{};
  std::vector<PreparedProgramStatusSlice> status_steps{};
  std::vector<PreparedProgramStatusSlice> telemetry_steps{};
  std::array<PreparedProgramStatusSlice, PreparedPipelineStepCapacity>
      status_ranges{};
  std::array<PreparedProgramStatusSlice, PreparedPipelineStepCapacity>
      telemetry_ranges{};
  std::array<VulkanPipelineWork, PreparedPipelineStepCapacity> template_work{};
  PreparedMemory recurrence_staging{};
  PreparedMemory transducer_staging{};
  std::vector<VulkanPipelineWork> transducer_work{};
  std::vector<std::uint64_t> transducer_occurrences{};
  std::uint64_t window_dispatches{};
  std::uint64_t window_gate_count{};
  std::uint64_t status_command_sources{};
  std::uint64_t telemetry_command_count{};
  std::uint64_t described_status_entry_count{};
  std::uint64_t encoded_work_command_count{};
  VulkanPipelineDescriptionCapacity description_capacity{};
};

[[nodiscard]] rund::AccelCheck PrepareVulkanPipelineAllocation(
    VulkanPipelinePreparation &);

[[nodiscard]] rund::AccelCheck PrepareVulkanPipelineDescriptions(
    VulkanPipelinePreparation &);

[[nodiscard]] rund::AccelCheck PrepareVulkanPipelineOccurrences(
    VulkanPipelinePreparation &);

[[nodiscard]] rund::AccelCheck PrepareVulkanPipelinePublication(
    VulkanPipelinePreparation &);

[[nodiscard]] rund::AccelCheck CaptureVulkanPipeline(
    VulkanPipelinePreparation &);

[[nodiscard]] rund::AccelCheck PrepareVulkanPipelineResidencyMemory(
    VulkanPipelinePreparation &);

#endif

} // namespace rund::node::accel::detail
