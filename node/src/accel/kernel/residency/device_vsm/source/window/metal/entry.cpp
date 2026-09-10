#include "internal.hpp"

#include "../../../../../../metal/range/source/algebra.hpp"

namespace rund::node::accel::detail::device_vsm_window_source {

bool emit_metal(const RangeExec &execution,
                const rund::kernel::ArtifactKey &key,
                const DeviceVsmWindowFusion &fusion,
                const DeviceVsmWindowRingPlan &ring,
                const DeviceVsmWindowMapSources &sources,
                std::string &source) noexcept {
  try {
    source.clear();
    if (ring.gpu_owned) {
      return metal::emit_ring(execution, key, fusion, ring, sources, source);
    }
    source += R"MSL(#include <metal_stdlib>
using namespace metal;
// artifact_variant=device_vsm

struct RangeParams {
  ulong input_count;
  ulong output_count;
  ulong window_size;
  ulong stride;
  ulong padding;
  ulong stage_element_count;
  ulong stage_aux_count;
  uint stage;
  uint reserved;
};

struct RundDeviceVsmConfig {
  ulong logical_elements;
  uint payload_elements;
  uint page_count;
  uint width;
  uint reserved;
};

)MSL";
    if (!append_metal_map_functions(source, fusion, sources)) {
      return false;
    }
    if (execution.saturating_sum()) {
      AppendMetalRangeSaturatingAlgebra(source);
    }
    const char *const type = metal::scalar_type(execution);
    source += "kernel void ";
    source += entry_name(key);
    source += "(\n    constant RangeParams& params [[buffer(0)]],\n";
    source += "    device const ";
    source += type;
    source += "* input [[buffer(1)]],\n    device ";
    source += type;
    source += R"MSL(* output [[buffer(2)]],
    constant RundDeviceVsmConfig& rund_vsm [[buffer(3)]],
    device atomic_uint* rund_vsm_result [[buffer(4)]],
    uint rund_local [[thread_index_in_threadgroup]],
    uint rund_worker [[threadgroup_position_in_grid]],
    uint rund_group_width [[threads_per_threadgroup]]) {
  for (uint rund_page = rund_worker; rund_page < rund_vsm.page_count;
       rund_page += rund_vsm.width) {
    const ulong rund_page_begin =
        ulong(rund_page) * ulong(rund_vsm.payload_elements);
    const ulong rund_remaining = params.output_count - rund_page_begin;
    const uint rund_page_elements =
        uint(min(rund_remaining, ulong(rund_vsm.payload_elements)));
)MSL";
    if (execution.uses_shared_halo()) {
      metal::append_shared(source, execution, type, fusion);
    } else {
      metal::append_direct(source, execution, type, fusion);
    }
    metal::append_counters(source);
    return !entry_name(key).empty();
  } catch (...) {
    source.clear();
    return false;
  }
}

} // namespace rund::node::accel::detail::device_vsm_window_source
