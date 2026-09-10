#pragma once

#include "../../../../../kernel/residency/persistent_sliding.hpp"
#include "../../../../../kernel/residency/schedule.hpp"
#include "../../../../../kernel/residency/window.hpp"
#include "../../../../command/model.hpp"
#include "../../../../timeline/model.hpp"

#include "../../../../adapter/buffer.hpp"
#include "../../../../adapter/pipeline.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanAdapter;
struct VulkanPipeline;

inline constexpr std::size_t VulkanResidencyWindowCapacity = 4u;
inline constexpr std::size_t VulkanResidencyCommandCapacity =
    ResidencyWindowLocalCapacity + 2u;

struct VulkanResidencyWindowBatch final {
  VulkanPipeline *pipeline{};
  std::array<VkCommandBuffer, VulkanResidencyCommandCapacity> commands{};
  std::size_t command_count{};
  VulkanTimelinePoint point{};
  std::uint64_t dispatch_count{};
  std::uint64_t control_count{};
  std::uint64_t reset_count{};
  std::uint64_t reset_bytes{};
  rund::AccelCheck admission{true, "ok"};
  bool signaled{};
};

struct VulkanResidencyScheduleRole final {
  VulkanPipeline *pipeline{};
  std::array<VkCommandBuffer, VulkanResidencyCommandCapacity> commands{};
  std::size_t command_count{};
  std::uint64_t dispatch_count{};
  std::uint64_t control_count{};
  std::uint64_t reset_count{};
  std::uint64_t reset_bytes{};
};

struct VulkanResidencyScheduleCell final {
  rund::AccelCheck admission{true, "ok"};
  std::uint64_t epoch{std::numeric_limits<std::uint64_t>::max()};
  bool signaled{};
};

struct VulkanResidencyScheduleRun final {
  std::mutex gate{};
  std::condition_variable ready{};
  BackendResidencyScheduleRequest request{};
  std::array<VulkanResidencyScheduleRole, ResidencyScheduleRoleCapacity>
      roles{};
  VulkanResidencyScheduleRole tail{};
  std::array<VulkanResidencyScheduleCell, ResidencyScheduleRoleCapacity>
      cells{};
  std::array<std::uint64_t, ResidencyScheduleRoleCapacity> ready_base{};
  std::uint64_t done_base{};
  std::uint32_t timeline_generation{};
  std::uint64_t release_count{};
  std::uint64_t signaled_count{};
  std::uint64_t inflight_peak{};
  std::uint64_t success_prefix{};
  std::uint64_t suppressed_first{};
  std::uint64_t suppressed_count{};
  std::uint64_t submit_begin_ns{};
  rund::AccelCheck first_failure{true, "ok"};
  rund::AccelCheck abort_failure{false, "accel_kernel_pipeline_invalid"};
  bool aborting{};
  bool active{};
  bool final_sent{};
  bool quarantined{};
};

struct VulkanResidencyPersistentRole final {
  VulkanPipeline *pipeline{};
  std::array<VkCommandBuffer, VulkanResidencyCommandCapacity + 1u> commands{};
  std::size_t command_count{};
  std::uint64_t dispatch_count{};
  std::uint64_t control_count{};
  std::uint64_t reset_count{};
  std::uint64_t reset_bytes{};
};

struct VulkanResidencyPersistentCell final {
  rund::AccelCheck admission{true, "ok"};
  std::uint64_t coordinate{std::numeric_limits<std::uint64_t>::max()};
  bool signaled{};
  bool waiting{};
  bool observed{};
  bool acknowledged{};
};

struct VulkanResidencyPersistentRun final {
  std::mutex gate{};
  PersistentResidencySlidingRequest request{};
  PersistentResidencySlidingControl *control{};
  std::array<VulkanResidencyPersistentRole, ResidencySlidingCapacity> roles{};
  VulkanResidencyPersistentRole tail{};
  std::array<VulkanResidencyPersistentCell, VulkanResidencyWindowCapacity>
      cells{};
  std::array<std::uint64_t, VulkanResidencyWindowCapacity> ready_base{};
  std::uint64_t done_base{};
  std::uint32_t timeline_generation{};
  std::uint64_t submit_begin_ns{};
  bool active{};
  bool final_sent{};
  bool quarantined{};
};

struct VulkanResidencyWindowRun final {
  std::mutex gate{};
  std::condition_variable ready{};
  BackendResidencyWindowRequest request{};
  std::array<VulkanResidencyWindowBatch, VulkanResidencyWindowCapacity>
      batches{};
  std::array<BackendResidencyWindowReceipt, VulkanResidencyWindowCapacity>
      receipts{};
  std::uint32_t timeline_generation{};
  std::size_t release_count{};
  std::size_t signaled_count{};
  std::size_t inflight_peak{};
  std::uint64_t submit_begin_ns{};
  rund::AccelCheck abort_failure{false, "accel_kernel_pipeline_invalid"};
  bool aborting{};
  bool active{};
  bool final_sent{};
  bool quarantined{};
};

struct VulkanResidencyStep final {
  VulkanCommand command{};
  std::uint64_t dispatch_count{};
  std::uint64_t control_count{};
  std::uint64_t reset_count{};
  std::uint64_t reset_bytes{};
  std::uint32_t declared_step{};
};

enum class VulkanResidencyMode : std::uint8_t {
  Direct,
  GraphStageDirect,
  GraphStageGeneratedIndirect,
  GraphStageSequence,
  GeneratedIndirectMap,
};

struct VulkanResidencyStatus final {
  bool valid{};
  bool present{};
  bool ready{};
  bool bounded{};
  bool quarantined{};
  bool adapter_quarantined{};
  VulkanResidencyMode mode{VulkanResidencyMode::Direct};
  std::uint32_t local_count{};
  std::uint32_t command_count{};
  std::uint32_t step_buffer_count{};
  bool prefix_buffer{};
  bool prefix_fence{};
  bool suffix_buffer{};
  bool arguments_buffer{};
  bool arguments_mapped{};
  const char *readiness_reason{"accel_kernel_pipeline_invalid"};
};

#endif

} // namespace rund::node::accel::detail
