#pragma once

#include <accel/check.hpp>

#include "../../kernel/callback.hpp"
#include "../command/model.hpp"
#include "../command/ring.hpp"
#include "buffer.hpp"
#include "pipeline.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include <vulkan/vulkan.h>
#endif

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanResidentState;
struct VulkanPipelineIndex;

struct VulkanAdapter {
  struct CommandSlot final {
    VulkanCommand command{};
    VkQueryPool timestamps{VK_NULL_HANDLE};
  };

  struct Pending final {
    VulkanCommandLease command{};
    VkFence external_fence = VK_NULL_HANDLE;
    KernelCompletion completion = nullptr;
    void *user = nullptr;
    std::uint64_t submitted_ns = 0u;
    VulkanBuffer staging{};
    std::shared_ptr<void> target{};
    std::uint64_t inflight = 0u;
    bool timestamp = true;
    bool external = false;
    bool force_device_lost = false;
  };
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkPhysicalDeviceMemoryProperties memory_properties{};
  VkDevice device = VK_NULL_HANDLE;
  VkQueue compute_queue = VK_NULL_HANDLE;
  std::uint32_t compute_queue_family = 0u;
  rund::kernel::ComputeCaps caps{};
  std::string device_name{};
  std::string driver_name{};
  std::string driver_info{};
  std::string glslang_validator_path{};
  std::string spirv_val_path{};
  std::uint32_t max_dispatch_groups = 0u;
  std::uint32_t dispatch_rows = 0u;
  VkDeviceSize storage_align = 1u;
  VkDeviceSize storage_limit = 1u;
  float timestamp_period_ns = 0.0F;
  std::uint32_t timestamp_valid_bits = 0u;
  bool timestamp_query_available = false;
  std::weak_ptr<void> owner_token{};
  std::mutex mutex;
  std::condition_variable command_cv;
  std::condition_variable host_readback_cv;
  std::size_t active_host_readbacks = 0u;
  std::mutex completion_mutex;
  std::condition_variable completion_cv;
  std::thread completion_thread{};
  std::array<Pending, kVulkanCommandCapacity> pending{};
  std::size_t pending_head = 0u;
  std::size_t pending_size = 0u;
  bool completion_stop = false;
  // Prepared Map jobs retain pipeline addresses for their full lifetime.
  std::deque<VulkanCachedPipeline> pipelines{};
  std::unique_ptr<VulkanPipelineIndex> pipeline_index{};
  std::deque<VulkanCollectivePipeline> collective_pipelines{};
  std::vector<VulkanCollectiveDescriptorLease> *active_descriptor_leases =
      nullptr;
  std::vector<VulkanBuffer> reusable_buffers{};
  std::uint64_t reusable_buffer_bytes = 0u;
  std::vector<VkDescriptorBufferInfo> descriptor_infos{};
  std::vector<VkWriteDescriptorSet> descriptor_writes{};
  std::unique_ptr<VulkanResidentState> resident{};
  std::array<CommandSlot, kVulkanCommandCapacity> commands{};
  VulkanCommandRing command_ring{};
  std::size_t recording_command = kInvalidVulkanCommand;
  // Borrowed encoding cursor valid only while mutex is held.
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  std::uint64_t dispatch_count = 0u;
  std::uint64_t command_submit_count = 0u;
  std::uint64_t command_inflight_peak = 0u;
  std::uint64_t command_capacity_rejection_count = 0u;
  std::uint64_t pipeline_compile_count = 0u;
  std::uint64_t pipeline_cache_hit_count = 0u;
  std::uint64_t shader_module_current = 0u;
  std::uint64_t shader_module_peak = 0u;
  std::uint64_t shader_module_create_count = 0u;
  std::uint64_t descriptor_pool_create_count = 0u;
  std::uint64_t descriptor_set_allocate_count = 0u;
  std::uint64_t descriptor_reuse_hit_count = 0u;
  std::uint64_t buffer_allocation_count = 0u;
  std::uint64_t buffer_reuse_hit_count = 0u;
  VulkanMemoryStats staging_memory{};
  std::uint64_t host_to_device_bytes = 0u;
  std::uint64_t device_to_host_bytes = 0u;
  std::uint64_t accel_kernel_ns = 0u;
  std::uint64_t accel_timestamp_count = 0u;
  const char *accel_timestamp_source = "unavailable";
  std::uint64_t shader_compile_ns = 0u;
  std::uint64_t spirv_compile_ns = 0u;
  std::uint64_t pipeline_create_ns = 0u;
  std::uint64_t descriptor_setup_ns = 0u;
  std::uint64_t command_submit_wait_ns = 0u;
  std::uint64_t readback_ns = 0u;
  std::atomic<bool> fault_device_lost_once{false};
  VulkanAdapter();
  VulkanAdapter(const VulkanAdapter &) = delete;
  VulkanAdapter &operator=(const VulkanAdapter &) = delete;
  ~VulkanAdapter();
};

[[nodiscard]] bool StartVulkanCompletionService(VulkanAdapter &adapter);
void StopVulkanCompletionService(VulkanAdapter &adapter) noexcept;

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)
} // namespace rund::node::accel::detail
