#include "internal.hpp"

#include <kernel/core/checked.hpp>

#include <cstring>

namespace rund::node::accel::detail::metal_device_vsm {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

DeviceVsmRearmResult
rearm(const std::shared_ptr<void> &lowering,
      const std::shared_ptr<const DeviceVsmProof> &proof) noexcept {
  DeviceVsmRearmResult out{};
  const auto fail = [&out](const char *const why) noexcept {
    out.check = {false, why};
    return out;
  };
  const std::shared_ptr<Owner> owner = owner_of(lowering);
  if (owner == nullptr || owner->magic != OwnerMagic || owner->proof != proof ||
      proof == nullptr || !device_vsm_proof_valid(*proof)) {
    return fail("accel_kernel_pipeline_invalid");
  }
  std::lock_guard lock{owner->gate};
  if (owner->adapter == nullptr) {
    return fail("compute_device_lost");
  }
  if (owner->in_flight) {
    return fail("compute_pipeline_busy");
  }
  if (!owner->submitted) {
    return fail("accel_kernel_pipeline_invalid");
  }
  std::lock_guard terminal_gate{owner->adapter->residency_terminal_gate};
  if (owner->adapter->residency_quarantined.load(std::memory_order_acquire)) {
    return fail("compute_device_lost");
  }
  const bool graph = proof->topology == DeviceVsmTopology::GraphResident;
  if (graph && (owner->graph_icb == nullptr ||
                owner->graph_pipeline != owner->pipeline ||
                owner->graph_digest != proof->graph_resident.digest ||
                !device_vsm_graph_resident_type_valid(
                    proof->graph_resident.type) ||
                proof->geometry.element_bytes !=
                    proof->graph_resident.type.element_bytes)) {
    return fail("accel_kernel_pipeline_invalid");
  }
  const bool graph_map =
      graph && device_vsm_page_map_active(proof->graph_resident.page_map);
  const bool graph_pointwise_map =
      proof->topology == DeviceVsmTopology::GraphPointwise &&
      device_vsm_page_map_active(proof->graph_pointwise.page_map);
  if (graph_map || graph_pointwise_map) {
    const DeviceVsmPageMap &page_map = graph_map
                                           ? proof->graph_resident.page_map
                                           : proof->graph_pointwise.page_map;
    id<MTLBuffer> map = (__bridge id<MTLBuffer>)owner->params.get();
    if (!device_vsm_page_map_valid(page_map, proof->geometry.page_count) ||
        map == nil || [map contents] == nullptr ||
        static_cast<std::uint64_t>([map length]) != DeviceVsmPageMapBytes ||
        static_cast<std::uint64_t>([map allocatedSize]) <
            DeviceVsmPageMapBytes) {
      return fail("accel_metal_buffer_unavailable");
    }
    const auto *const words =
        static_cast<const std::uint32_t *>([map contents]);
    for (std::size_t index = 0u; index < page_map.words.size(); ++index) {
      if (words[index] != page_map.words[index]) {
        return fail("accel_metal_buffer_unavailable");
      }
    }
  }
  const bool window_ring = proof->topology == DeviceVsmTopology::Window &&
                           proof->window.ring.gpu_owned;
  DeviceVsmWindowRingPlan window_ring_plan{};
  DeviceVsmWindowRingResult window_ring_result{};
  std::array<
      std::array<MetalResidentBufferResult, DeviceVsmGraphResidentBankCapacity>,
      DeviceVsmGraphResidentPhysicalCapacity>
      fresh_owners{};
  std::array<MetalResidentBufferResult, DeviceVsmResidentCapacity>
      fresh_endpoints{};
  std::array<MetalResidentBufferResult, DeviceVsmResidentCapacity>
      fresh_residents{};
  if (graph &&
      (owner->graph.ready == 0u ||
       owner->graph.owner_count != proof->graph_resident.owner_binding_count ||
       owner->graph.endpoint_count != proof->residents.count)) {
    return fail("accel_kernel_pipeline_invalid");
  }
  if (graph) {
    for (std::size_t slot = 0u; slot < owner->graph.owner_count; ++slot) {
      const auto &source = proof->graph_resident.owners[slot];
      for (std::size_t bank = 0u; bank < DeviceVsmGraphResidentBankCapacity;
           ++bank) {
        MetalResidentBufferResult current = LookupMetalResidentBuffer(
            owner->device, source.refs[bank], source.handles[bank]);
        const auto &old = owner->graph.owners[slot][bank];
        if (!current.check.ok || current.device_buffer == nullptr ||
            !same_ref(current.ref, old.ref) ||
            !device_vsm_graph_resident_same_object(current.handle,
                                                   old.handle) ||
            current.device_buffer != old.device_buffer) {
          return fail("accel_metal_buffer_unavailable");
        }
        fresh_owners[slot][bank] = std::move(current);
      }
    }
    for (std::size_t endpoint = 0u; endpoint < owner->graph.endpoint_count;
         ++endpoint) {
      const auto &source = proof->residents.rows[endpoint];
      MetalResidentBufferResult current = LookupMetalResidentBuffer(
          owner->device, source.backing, source.handle);
      const auto &old = owner->graph.endpoints[endpoint];
      if (!current.check.ok || current.device_buffer == nullptr ||
          !same_ref(current.ref, old.ref) ||
          !device_vsm_graph_resident_same_object(current.handle, old.handle) ||
          current.device_buffer != old.device_buffer) {
        return fail("accel_metal_buffer_unavailable");
      }
      fresh_endpoints[endpoint] = std::move(current);
    }
  }
  id<MTLBuffer> result = (__bridge id<MTLBuffer>)owner->result.get();
  if (result == nil || [result contents] == nullptr) {
    return fail("accel_metal_buffer_unavailable");
  }
  if (window_ring) {
    if (!device_vsm_window_ring_plan_expected(proof->geometry,
                                              window_ring_plan) ||
        proof->window.ring != window_ring_plan ||
        !device_vsm_window_ring_result_expected(
            proof->geometry, window_ring_plan, window_ring_result) ||
        owner->resident_count != 2u || proof->residents.count != 2u ||
        proof->residents.input_count != 1u ||
        proof->residents.output_count != 1u || owner->config == nullptr ||
        owner->ring_state == nullptr ||
        owner->ring_scratch_count != DeviceVsmWindowRingSlotCount) {
      return fail("accel_kernel_pipeline_invalid");
    }
    id<MTLBuffer> config = (__bridge id<MTLBuffer>)owner->config.get();
    id<MTLBuffer> state = (__bridge id<MTLBuffer>)owner->ring_state.get();
    if (config == nil || [config contents] == nullptr ||
        static_cast<std::uint64_t>([config allocatedSize]) <
            window_ring_plan.config_bytes ||
        state == nil || [state contents] == nullptr ||
        static_cast<std::uint64_t>([state allocatedSize]) <
            window_ring_plan.state_bytes ||
        window_ring_plan.scratch_bytes % DeviceVsmWindowRingSlotCount != 0u) {
      return fail("accel_metal_buffer_unavailable");
    }
    const std::uint64_t scratch_bytes =
        window_ring_plan.scratch_bytes / DeviceVsmWindowRingSlotCount;
    for (std::size_t index = 0u; index < owner->resident_count; ++index) {
      const auto &source = proof->residents.rows[index];
      MetalResidentBufferResult current = LookupMetalResidentBuffer(
          owner->device, source.backing, source.handle);
      const auto &old = owner->residents[index];
      if (!current.check.ok || current.device_buffer == nullptr ||
          !same_ref(current.ref, old.ref) ||
          !device_vsm_graph_resident_same_object(current.handle, old.handle) ||
          current.device_buffer != old.device_buffer) {
        return fail("accel_metal_buffer_unavailable");
      }
      fresh_residents[index] = std::move(current);
    }
    for (std::size_t index = 0u; index < owner->ring_scratch_count; ++index) {
      id<MTLBuffer> scratch =
          (__bridge id<MTLBuffer>)owner->ring_scratch[index].get();
      if (scratch == nil || [scratch contents] == nullptr ||
          static_cast<std::uint64_t>([scratch allocatedSize]) < scratch_bytes) {
        return fail("accel_metal_buffer_unavailable");
      }
    }
  }
  const bool ring = proof->topology == DeviceVsmTopology::Pointwise ||
                    proof->topology == DeviceVsmTopology::GraphPointwise;
  std::uint64_t state_bytes = 0u;
  std::uint64_t scratch_bytes = 0u;
  std::uint64_t region_bytes = 0u;
  std::uint64_t expected_scratch_bytes = 0u;
  id<MTLBuffer> ring_state = (__bridge id<MTLBuffer>)owner->ring_state.get();
  if (ring && (!device_vsm_ring_storage_expected(proof->geometry, proof->width,
                                                 proof->residents.count,
                                                 state_bytes, scratch_bytes) ||
               !::rund::kernel::checked::mul(
                   proof->width, proof->geometry.payload_bytes, region_bytes) ||
               !::rund::kernel::checked::mul(region_bytes,
                                            proof->residents.count,
                                            expected_scratch_bytes) ||
               scratch_bytes != expected_scratch_bytes ||
               ring_state == nil || [ring_state contents] == nullptr ||
               owner->ring_scratch_count != proof->residents.count)) {
    return fail("accel_metal_buffer_unavailable");
  }
  out.mutated = true;
  if (graph_map || graph_pointwise_map) {
    const DeviceVsmPageMap &page_map = graph_map
                                           ? proof->graph_resident.page_map
                                           : proof->graph_pointwise.page_map;
    id<MTLBuffer> map = (__bridge id<MTLBuffer>)owner->params.get();
    auto *const words = static_cast<std::uint32_t *>([map contents]);
    for (std::size_t index = 0u; index < page_map.words.size(); ++index) {
      words[index] = page_map.words[index];
    }
  }
  std::memset([result contents], 0,
              DeviceVsmResultWordCount * sizeof(std::uint32_t));
  static_cast<std::uint32_t *>([result contents])[DeviceVsmFailedPageWord] =
      DeviceVsmNoFailedPage;
  if (ring) {
    std::memset([ring_state contents], 0,
                static_cast<std::size_t>(state_bytes));
    for (std::size_t index = 0u; index < owner->ring_scratch_count; ++index) {
      id<MTLBuffer> scratch =
          (__bridge id<MTLBuffer>)owner->ring_scratch[index].get();
      if (scratch == nil || [scratch contents] == nullptr) {
        return fail("accel_metal_buffer_unavailable");
      }
      std::memset([scratch contents], 0,
                  static_cast<std::size_t>(region_bytes));
    }
  }
  if (window_ring) {
    for (std::size_t index = 0u; index < owner->resident_count; ++index) {
      owner->residents[index] = std::move(fresh_residents[index]);
    }
    id<MTLBuffer> state = (__bridge id<MTLBuffer>)owner->ring_state.get();
    std::memset([state contents], 0,
                static_cast<std::size_t>(window_ring_plan.state_bytes));
    const std::size_t scratch_bytes = static_cast<std::size_t>(
        window_ring_plan.scratch_bytes / DeviceVsmWindowRingSlotCount);
    for (std::size_t index = 0u; index < owner->ring_scratch_count; ++index) {
      id<MTLBuffer> scratch =
          (__bridge id<MTLBuffer>)owner->ring_scratch[index].get();
      std::memset([scratch contents], 0, scratch_bytes);
    }
    id<MTLBuffer> config = (__bridge id<MTLBuffer>)owner->config.get();
    auto *const base = static_cast<std::uint8_t *>([config contents]);
    const WindowRingConfig item{
        .logical_elements = static_cast<std::uint32_t>(
            proof->geometry.logical_bytes / proof->geometry.element_bytes),
        .payload_elements = window_ring_plan.payload_elements,
        .page_count = window_ring_plan.page_count,
        .width = proof->width,
        .element_words = 1u,
        .epoch = 0u,
        .slot = 0u,
        .phase = 0u,
        .frame_elements = window_ring_plan.frame_elements,
        .halo_elements = window_ring_plan.halo_elements,
    };
    std::memcpy(base, &item, sizeof(item));
  }
  if (graph) {
    for (std::size_t slot = 0u; slot < owner->graph.owner_count; ++slot) {
      for (std::size_t bank = 0u; bank < DeviceVsmGraphResidentBankCapacity;
           ++bank) {
        owner->graph.owners[slot][bank] = std::move(fresh_owners[slot][bank]);
      }
    }
    for (std::size_t endpoint = 0u; endpoint < owner->graph.endpoint_count;
         ++endpoint) {
      owner->graph.endpoints[endpoint] = std::move(fresh_endpoints[endpoint]);
    }
    GraphResidentTable table{};
    table.logical_elements =
        proof->geometry.logical_bytes / proof->geometry.element_bytes;
    table.payload_elements = static_cast<std::uint32_t>(
        proof->geometry.payload_bytes / proof->geometry.element_bytes);
    table.page_count = static_cast<std::uint32_t>(proof->geometry.page_count);
    table.element_words = static_cast<std::uint32_t>(
        proof->graph_resident.type.element_bytes / sizeof(std::uint32_t));
    table.element_bytes = proof->graph_resident.type.element_bytes;
    table.valid = 1u;
    table.owner_count = owner->graph.owner_count;
    table.controller = device_vsm_graph_controller(proof->graph_wavefront);
    for (std::size_t slot = 0u; slot < owner->graph.owner_count; ++slot) {
      table.owner_valid[slot] = 1u;
      for (std::size_t bank = 0u; bank < DeviceVsmGraphResidentBankCapacity;
           ++bank) {
        table.owner_generations[slot * DeviceVsmGraphResidentBankCapacity +
                                bank] =
            proof->graph_resident.owners[slot].buffer_generations[bank];
      }
    }
    id<MTLBuffer> table_buffer =
        (__bridge id<MTLBuffer>)owner->graph.table.get();
    if (table_buffer == nil || [table_buffer contents] == nullptr) {
      return fail("accel_metal_graph_table_unavailable");
    }
    std::memcpy([table_buffer contents], &table, sizeof(table));
  }
  owner->submitted = false;
  out.check = {true, "ok"};
  return out;
}

#endif

} // namespace rund::node::accel::detail::metal_device_vsm
