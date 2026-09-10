#include "internal.hpp"

#include <cstring>

namespace rund::node::accel::detail {

MetalDeviceVsmPreparation PrepareMetalDeviceVsm(
    const rund::AccelDevice &pick,
    const std::shared_ptr<const DeviceVsmProof> &proof) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  using namespace metal_device_vsm;
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr) {
    return MetalDeviceVsmPreparation{
        .capability = {.check = {false, "accel_metal_unavailable"}}};
  }
  if (adapter->device == nullptr) {
    return MetalDeviceVsmPreparation{
        .capability = {.check = {false, "accel_metal_device_unavailable"}}};
  }
  if (adapter->queue == nullptr) {
    return MetalDeviceVsmPreparation{
        .capability = {.check = {false, "accel_metal_queue_unavailable"}}};
  }
  if (proof == nullptr || !device_vsm_proof_valid(*proof)) {
    return MetalDeviceVsmPreparation{
        .capability = {.check = {false, "accel_kernel_pipeline_invalid"}}};
  }
  if (proof->plan.api != rund::kernel::ComputeApi::Metal) {
    return MetalDeviceVsmPreparation{
        .capability = {.check = {false, "compute_backend_mismatch"}}};
  }
  std::shared_ptr<Owner> owner = make_owner();
  if (owner == nullptr) {
    return MetalDeviceVsmPreparation{
        .capability = {.check = {false, "compute_pipeline_capacity"}}};
  }
  owner->adapter = adapter;
  owner->adapter_owner = pick.owner;
  owner->device = pick;
  owner->proof = proof;

  prepare::Shape shape{};
  const rund::AccelCheck shape_check =
      prepare::ValidateShapeAndBindings(*owner, pick, *proof, shape);
  if (!shape_check.ok) {
    return MetalDeviceVsmPreparation{.capability = {.check = shape_check}};
  }
  prepare::ResourcePlan resource_plan{};
  const rund::AccelCheck resource_check = prepare::AllocateResources(
      *owner, *adapter, *proof, shape, resource_plan);
  if (!resource_check.ok) {
    return MetalDeviceVsmPreparation{.capability = {.check = resource_check}};
  }

  const bool graph = shape.graph;
  const bool map_storage = shape.graph_map || shape.graph_pointwise_map;
  const bool ring_storage = shape.ring_storage;
  const NSUInteger parameter_bytes =
      static_cast<NSUInteger>(resource_plan.parameter_bytes);
  const std::uint64_t ring_state_bytes = resource_plan.ring_state_bytes;
  const std::uint64_t ring_region_bytes = resource_plan.ring_region_bytes;
  for (std::size_t index = 0u; index < owner->resident_count; ++index) {
    if (!owner->residents[index].check.ok) {
      return MetalDeviceVsmPreparation{
          .capability = {.check = owner->residents[index].check}};
    }
    if (owner->residents[index].device_buffer == nullptr) {
      return MetalDeviceVsmPreparation{
          .capability = {.check = {false, "accel_metal_buffer_unavailable"}}};
    }
  }
  if (graph) {
    for (std::size_t endpoint = 0u; endpoint < owner->graph.endpoint_count;
         ++endpoint) {
      if (!owner->graph.endpoints[endpoint].check.ok ||
          owner->graph.endpoints[endpoint].device_buffer == nullptr) {
        return MetalDeviceVsmPreparation{
            .capability = {.check = owner->graph.endpoints[endpoint].check}};
      }
    }
    for (std::size_t slot = 0u; slot < owner->graph.owner_count; ++slot) {
      for (std::size_t bank = 0u; bank < DeviceVsmGraphResidentBankCapacity;
           ++bank) {
        const auto &binding = owner->graph.owners[slot][bank];
        if (!binding.check.ok || binding.device_buffer == nullptr) {
          return MetalDeviceVsmPreparation{
              .capability = {.check = binding.check}};
        }
      }
    }
  }
  if (owner->pipeline == nullptr) {
    return MetalDeviceVsmPreparation{
        .capability = {.check = {false, MetalLastError(adapter)}}};
  }
  id<MTLComputePipelineState> pipeline =
      (__bridge id<MTLComputePipelineState>)owner->pipeline.get();
  const std::uint32_t required_width =
      proof->topology == DeviceVsmTopology::Window
          ? proof->window.workgroup_width
      : proof->topology == DeviceVsmTopology::GraphPointwise
          ? proof->graph_pointwise.workgroup_width
      : proof->topology == DeviceVsmTopology::GraphMapReduce
          ? proof->graph_map_reduce.workgroup_width
      : proof->topology == DeviceVsmTopology::Scan ? proof->scan.workgroup_width
      : proof->topology == DeviceVsmTopology::Reduce
          ? proof->reduce.workgroup_width
          : 0u;
  if (required_width != 0u &&
      [pipeline maxTotalThreadsPerThreadgroup] < required_width) {
    return MetalDeviceVsmPreparation{
        .capability = {.check = {false, "compute_pipeline_capacity"}}};
  }
  if (owner->params == nullptr || (!graph && owner->config == nullptr) ||
      (graph && owner->graph.table == nullptr) || owner->result == nullptr ||
      (ring_storage && owner->ring_state == nullptr)) {
    return MetalDeviceVsmPreparation{
        .capability = {.check = {false, "accel_metal_buffer_unavailable"}}};
  }
  id<MTLBuffer> params = (__bridge id<MTLBuffer>)owner->params.get();
  id<MTLBuffer> config =
      graph ? nil : (__bridge id<MTLBuffer>)owner->config.get();
  id<MTLBuffer> result = (__bridge id<MTLBuffer>)owner->result.get();
  id<MTLBuffer> ring_state = (__bridge id<MTLBuffer>)owner->ring_state.get();
  if ([params contents] == nullptr ||
      (!graph && [config contents] == nullptr) ||
      [result contents] == nullptr ||
      (ring_storage && [ring_state contents] == nullptr)) {
    return {};
  }
  if (map_storage &&
      (static_cast<std::uint64_t>([params length]) != DeviceVsmPageMapBytes ||
       static_cast<std::uint64_t>([params allocatedSize]) <
           DeviceVsmPageMapBytes)) {
    return MetalDeviceVsmPreparation{
        .capability = {.check = {false, "device_vsm_graph_binding_invalid"}}};
  }
  if (shape.window_ring && static_cast<std::uint64_t>([config allocatedSize]) <
                               shape.window_ring_plan.config_bytes) {
    return MetalDeviceVsmPreparation{
        .capability = {.check = {false, "compute_pipeline_capacity"}}};
  }
  if (ring_storage) {
    for (std::size_t index = 0u; index < owner->ring_scratch_count; ++index) {
      id<MTLBuffer> scratch =
          (__bridge id<MTLBuffer>)owner->ring_scratch[index].get();
      if (scratch == nil || [scratch contents] == nullptr) {
        return {};
      }
    }
  }
  if (graph && !prepare::RecordGraph(*owner, *adapter)) {
    return MetalDeviceVsmPreparation{
        .capability = {.check = {false, "accel_metal_command_unavailable"}}};
  }
  std::memset([params contents], 0, parameter_bytes);
  if (shape.graph_map || shape.graph_pointwise_map) {
    const DeviceVsmPageMap &page_map = shape.graph_map
                                           ? proof->graph_resident.page_map
                                           : proof->graph_pointwise.page_map;
    auto *const words = static_cast<std::uint32_t *>([params contents]);
    for (std::size_t index = 0u; index < page_map.words.size(); ++index) {
      words[index] = page_map.words[index];
    }
  } else if (proof->parameter_bytes != 0u) {
    std::memcpy([params contents], proof->parameters,
                static_cast<std::size_t>(proof->parameter_bytes));
  }
  const Config configuration{
      .logical_elements =
          proof->geometry.logical_bytes / proof->geometry.element_bytes,
      .payload_elements = static_cast<std::uint32_t>(
          proof->geometry.payload_bytes / proof->geometry.element_bytes),
      .page_count = static_cast<std::uint32_t>(proof->geometry.page_count),
      .width = proof->width,
      .element_words = static_cast<std::uint32_t>(
          proof->geometry.element_bytes / sizeof(std::uint32_t)),
  };
  if (!graph && !shape.window_ring) {
    std::memcpy([config contents], &configuration, sizeof(configuration));
  }
  if (shape.window_ring) {
    std::byte *const base = static_cast<std::byte *>([config contents]);
    const WindowRingConfig item{
        .logical_elements = static_cast<std::uint32_t>(
            proof->geometry.logical_bytes / proof->geometry.element_bytes),
        .payload_elements = shape.window_ring_plan.payload_elements,
        .page_count = shape.window_ring_plan.page_count,
        .width = proof->width,
        .element_words = 1u,
        .epoch = 0u,
        .slot = 0u,
        .phase = 0u,
        .frame_elements = shape.window_ring_plan.frame_elements,
        .halo_elements = shape.window_ring_plan.halo_elements,
    };
    std::memcpy(base, &item, sizeof(item));
  }
  std::memset([result contents], 0,
              DeviceVsmResultWordCount * sizeof(std::uint32_t));
  static_cast<std::uint32_t *>([result contents])[DeviceVsmFailedPageWord] =
      DeviceVsmNoFailedPage;
  if (ring_storage) {
    std::memset([ring_state contents], 0,
                static_cast<std::size_t>(ring_state_bytes));
    for (std::size_t index = 0u; index < owner->ring_scratch_count; ++index) {
      id<MTLBuffer> scratch =
          (__bridge id<MTLBuffer>)owner->ring_scratch[index].get();
      std::memset([scratch contents], 0,
                  static_cast<std::size_t>(ring_region_bytes));
    }
  }
  std::uint64_t retained =
      sizeof(Owner) + static_cast<std::uint64_t>([params allocatedSize]) +
      static_cast<std::uint64_t>([config allocatedSize]) +
      static_cast<std::uint64_t>([result allocatedSize]) +
      (ring_storage ? static_cast<std::uint64_t>([ring_state allocatedSize])
                    : 0u);
  if (graph) {
    retained += static_cast<std::uint64_t>(
        [(__bridge id<MTLBuffer>)owner->graph.table.get() allocatedSize]);
    retained += static_cast<std::uint64_t>(
        [(__bridge id<MTLIndirectCommandBuffer>)owner->graph_icb.get()
            allocatedSize]);
  }
  if (ring_storage) {
    for (std::size_t index = 0u; index < owner->ring_scratch_count; ++index) {
      retained += static_cast<std::uint64_t>(
          [(__bridge id<MTLBuffer>)owner->ring_scratch[index].get()
              allocatedSize]);
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
  return MetalDeviceVsmPreparation{
      .capability = owner->capability,
      .lowering = owner,
      .submit = submit,
      .rearm = rearm,
  };
#else
  (void)pick;
  (void)proof;
  return {};
#endif
}

} // namespace rund::node::accel::detail
