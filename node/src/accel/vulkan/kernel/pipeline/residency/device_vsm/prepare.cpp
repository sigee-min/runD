#include "../../../../adapter/error.hpp"
#include "../../../../adapter/access.hpp"
#include "../../../../buffer/access.hpp"
#include "../../../../buffer/create.hpp"

#include "prepare/local.hpp"

#include <array>
#include <limits>

namespace rund::node::accel::detail::vulkan_device_vsm {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace {

[[nodiscard]] bool cleared_buffer(VulkanAdapter &adapter,
                                  const std::uint64_t bytes,
                                  ScopedBuffer &out) noexcept {
  if (bytes == 0u || bytes > std::numeric_limits<VkDeviceSize>::max()) {
    return false;
  }
  VulkanBuffer value{};
  const VkDeviceSize size = static_cast<VkDeviceSize>(bytes);
  if (!CreateVulkanBuffer(adapter, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          value)) {
    return false;
  }
  out = ScopedBuffer{adapter, value, size};
  return ClearVulkanBuffer(out.buffer, size);
}

} // namespace

#endif

} // namespace rund::node::accel::detail::vulkan_device_vsm

namespace rund::node::accel::detail {

namespace {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
static constexpr char AdapterProofApiInvalid[] =
    "device_vsm_adapter_proof_api_invalid";
static constexpr char GraphBindingInvalid[] =
    "device_vsm_graph_binding_invalid";
static constexpr char ParameterBufferAllocationFailed[] =
    "device_vsm_parameter_buffer_allocation_failed";
static constexpr char ResultBufferAllocationFailed[] =
    "device_vsm_result_buffer_allocation_failed";
static constexpr char RingStateAllocationFailed[] =
    "device_vsm_ring_state_allocation_failed";
static constexpr char RingScratchAllocationFailed[] =
    "device_vsm_ring_scratch_allocation_failed";
#else
static constexpr char SdkUnavailable[] = "device_vsm_sdk_unavailable";
#endif

[[nodiscard]] VulkanDeviceVsmPreparation
reject(const char *const reason) noexcept {
  return VulkanDeviceVsmPreparation{.capability = {.check = {false, reason}}};
}

} // namespace

VulkanDeviceVsmPreparation PrepareVulkanDeviceVsm(
    const rund::AccelDevice &pick,
    const std::shared_ptr<const DeviceVsmProof> &proof) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using namespace vulkan_device_vsm;
  VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr || proof == nullptr ||
      !device_vsm_proof_valid(*proof) ||
      proof->plan.api != rund::kernel::ComputeApi::Vulkan) {
    return reject(AdapterProofApiInvalid);
  }
  std::shared_ptr<Owner> owner = make_owner();
  if (owner == nullptr) {
    return VulkanDeviceVsmPreparation{
        .capability = {.check = {false, "compute_pipeline_capacity"}}};
  }
  owner->adapter = adapter;
  owner->adapter_owner = pick.owner;
  owner->device = pick;
  owner->proof = proof;

  bool ring = false;
  bool window_ring = false;
  bool ring_storage = false;
  bool graph = false;
  bool graph_map = false;
  bool graph_pointwise_map = false;
  DeviceVsmWindowRingPlan window_ring_plan{};
  std::uint64_t ring_state_bytes = 0u;
  std::uint64_t ring_scratch_bytes = 0u;
  std::uint64_t ring_region_bytes = 0u;
  const rund::AccelCheck shape = prepare::ValidateShape(
      *proof, ring, window_ring, ring_storage, graph, graph_map,
      graph_pointwise_map, window_ring_plan, ring_state_bytes,
      ring_scratch_bytes, ring_region_bytes);
  if (!shape.ok) {
    return reject(shape.reason);
  }
  const rund::AccelCheck residents =
      prepare::BindResidents(pick, *proof, *owner);
  if (!residents.ok) {
    return reject(residents.reason);
  }
  if (graph) {
    const rund::AccelCheck graph_preparation =
        prepare::PrepareGraph(pick, *proof, *owner);
    if (!graph_preparation.ok) {
      return reject(graph_preparation.reason);
    }
  }

  std::uint32_t descriptor_count = 0u;
  if (!descriptor_count_for(graph, window_ring, ring, owner->graph.owner_count,
                            owner->graph.endpoint_count, proof->residents.count,
                            descriptor_count)) {
    return reject(DescriptorCountOverflowReason);
  }
  std::lock_guard adapter_lock{adapter->mutex};
  SetVulkanLastError(*adapter, "device_vsm_pipeline_allocation_failed");
  owner->pipeline =
      AcquireVulkanCollectivePipeline(*adapter, descriptor_count,
                                      graph         ? 0u
                                      : window_ring ? sizeof(WindowRingDispatch)
                                                    : sizeof(Dispatch),
                                      proof->plan, *proof->artifact);
  if (owner->pipeline == nullptr) {
    return reject(VulkanLastError(adapter));
  }
  std::array<std::uint32_t, DeviceVsmResultWordCount> zero{};
  zero[DeviceVsmFailedPageWord] = DeviceVsmNoFailedPage;
  const bool map_storage = graph_map || graph_pointwise_map;
  const void *const parameter_data =
      graph_map             ? static_cast<const void *>(
                                  proof->graph_resident.page_map.words.data())
      : graph_pointwise_map ? static_cast<const void *>(
                                  proof->graph_pointwise.page_map.words.data())
                            : static_cast<const void *>(proof->parameters);
  const std::uint64_t parameter_bytes =
      map_storage ? DeviceVsmPageMapBytes : proof->parameter_bytes;
  if (!MakeHostBuffer(*adapter, parameter_data, parameter_bytes,
                      owner->params) ||
      (map_storage &&
       (owner->params.buffer.bytes != DeviceVsmPageMapBytes ||
        owner->params.buffer.capacity_bytes < DeviceVsmPageMapBytes ||
        owner->params.buffer.mapped == nullptr))) {
    return reject(ParameterBufferAllocationFailed);
  }
  if (!MakeHostBuffer(*adapter, zero.data(), sizeof(zero), owner->result)) {
    return reject(ResultBufferAllocationFailed);
  }
  if (window_ring) {
    ring_state_bytes = window_ring_plan.state_bytes;
    ring_region_bytes =
        window_ring_plan.scratch_bytes / DeviceVsmWindowRingSlotCount;
  }
  if (ring_storage &&
      !cleared_buffer(*adapter, ring_state_bytes, owner->ring_state)) {
    return reject(RingStateAllocationFailed);
  }
  if (ring_storage) {
    owner->ring_scratch_count =
        window_ring ? DeviceVsmWindowRingSlotCount : proof->residents.count;
    for (std::size_t index = 0u; index < owner->ring_scratch_count; ++index) {
      if (!cleared_buffer(*adapter, ring_region_bytes,
                          owner->ring_scratch[index])) {
        return reject(RingScratchAllocationFailed);
      }
    }
  }
  if (graph) {
    if (!prepare::FillGraphBindings(*owner)) {
      return reject(GraphBindingInvalid);
    }
    const rund::AccelCheck recorded = prepare::RecordGraph(*owner);
    if (!recorded.ok) {
      return reject(recorded.reason);
    }
  }
  std::uint64_t retained =
      sizeof(Owner) + owner->params.buffer.allocated_bytes +
      (graph ? owner->graph_table.buffer.allocated_bytes : 0u) +
      owner->result.buffer.allocated_bytes +
      (ring_storage ? owner->ring_state.buffer.allocated_bytes : 0u);
  if (graph) {
    retained += sizeof(GraphResidentState);
  }
  if (ring_storage) {
    for (std::size_t index = 0u; index < owner->ring_scratch_count; ++index) {
      retained += owner->ring_scratch[index].buffer.allocated_bytes;
    }
  }
  owner->capability = DeviceVsmCapability{
      .check = {true, "ok"},
      .retained_bytes = retained,
      .transient_bytes = 0u,
      .width = proof->width,
      .gpu_addressable_backing = true,
      .device_generated_recurrence = true,
      .fixed_native_storage = true,
      .fixed_common_storage = proof->fixed_common_storage,
      .one_native_submit = true,
      .host_service_turns_zero = true,
      .host_epoch_callbacks_zero = true,
      .aggregate_terminal_once = true,
      .bounded_page_io = true,
      .physical_ring_storage = true,
  };
  return VulkanDeviceVsmPreparation{
      .capability = owner->capability,
      .lowering = owner,
      .submit = submit,
      .rearm = rearm,
  };
#else
  (void)pick;
  (void)proof;
  return reject(SdkUnavailable);
#endif
}

} // namespace rund::node::accel::detail
