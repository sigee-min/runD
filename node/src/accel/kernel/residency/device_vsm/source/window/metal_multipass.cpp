#include "internal.hpp"

namespace rund::node::accel::detail::device_vsm_window_source {
namespace {

void append_increment(std::string &source) {
  source += R"MSL(    for (uint word = 0u; word < 6u; ++word) {
      atomic_fetch_add_explicit(&rund_vsm_result[word], 1u,
                                memory_order_relaxed);
    }
    const uint rund_first_page = page == 0u ? 0u : page - 1u;
    const uint rund_last_page = min(page + 1u, rund_vsm.page_count - 1u);
    const uint rund_page_elements = uint(end - begin);
    const uint rund_footprint =
        3u * (page + 1u) + 5u * rund_first_page + 7u * rund_last_page +
        11u * rund_page_elements;
    if (page + 1u < rund_vsm.page_count) {
      atomic_fetch_add_explicit(&rund_vsm_result[8], 1u,
                                memory_order_relaxed);
    }
    atomic_fetch_add_explicit(&rund_vsm_result[9], rund_footprint,
                              memory_order_relaxed);
)MSL";
}

void append_prefix(std::string &source, const DeviceVsmWindowFusion &fusion) {
  source += R"MSL(  uint prefix = 0u;
  for (ulong index = 0ul; index < params.input_count; ++index) {
    const uint sample = )MSL";
  append_map_chain_expression(source, fusion, MapSlot::Before, "input[index]",
                              "index");
  source += R"MSL(;
    prefix += sample;
    input[index] = prefix;
  }
  for (uint page = 0u; page < rund_vsm.page_count; ++page) {
    const ulong begin = ulong(page) * ulong(rund_vsm.payload_elements);
    const ulong end = min(begin + ulong(rund_vsm.payload_elements),
                          params.output_count);
    for (ulong index = begin; index < end; ++index) {
      const ulong anchor = index * params.stride;
      const ulong left = anchor < params.padding ? 0ul
                                                 : anchor - params.padding;
      const ulong right_width = params.window_size - params.padding;
      const ulong right =
          anchor >= params.input_count
              ? params.input_count - 1ul
              : (right_width >= params.input_count - anchor
                     ? params.input_count - 1ul
                     : anchor + right_width - 1ul);
      uint value = input[right];
      if (left != 0ul) { value -= input[left - 1ul]; }
      output[index] = )MSL";
  append_map_chain_expression(source, fusion, MapSlot::After, "value", "index");
  source += R"MSL(;
    }
)MSL";
  append_increment(source);
  source += "  }\n";
}

void append_extreme(std::string &source, const RangeExec &execution,
                    const DeviceVsmWindowFusion &fusion) {
  const char *const combine =
      execution.operation() == RangeOp::Minimum ? "min" : "max";
  source += R"MSL(  const ulong block_width = params.window_size;
  for (ulong block = 0ul; block < params.input_count; block += block_width) {
    const ulong end = min(block + block_width, params.input_count);
    uint value = )MSL";
  append_map_chain_expression(source, fusion, MapSlot::Before,
                              "input[end - 1ul]", "end - 1ul");
  source += R"MSL(;
    output[end - 1ul] = value;
    for (ulong cursor = end - 1ul; cursor > block; --cursor) {
      const uint sample = )MSL";
  append_map_chain_expression(source, fusion, MapSlot::Before,
                              "input[cursor - 1ul]", "cursor - 1ul");
  source += ";\n      value = ";
  source += combine;
  source += R"MSL((sample, value);
      output[cursor - 1ul] = value;
    }
  }
  for (ulong block = 0ul; block < params.input_count; block += block_width) {
    const ulong end = min(block + block_width, params.input_count);
    uint value = )MSL";
  append_map_chain_expression(source, fusion, MapSlot::Before, "input[block]",
                              "block");
  source += R"MSL(;
    input[block] = value;
    for (ulong cursor = block + 1ul; cursor < end; ++cursor) {
      const uint sample = )MSL";
  append_map_chain_expression(source, fusion, MapSlot::Before, "input[cursor]",
                              "cursor");
  source += ";\n      value = ";
  source += combine;
  source += R"MSL((value, sample);
      input[cursor] = value;
    }
  }
  for (uint page_cursor = rund_vsm.page_count; page_cursor != 0u;
       --page_cursor) {
    const uint page = page_cursor - 1u;
    const ulong begin = ulong(page) * ulong(rund_vsm.payload_elements);
    const ulong end = min(begin + ulong(rund_vsm.payload_elements),
                          params.output_count);
    for (ulong cursor = end; cursor > begin; --cursor) {
      const ulong index = cursor - 1ul;
      const ulong anchor = index * params.stride;
      const ulong left = anchor < params.padding ? 0ul
                                                 : anchor - params.padding;
      const ulong right_width = params.window_size - params.padding;
      const ulong right =
          anchor >= params.input_count
              ? params.input_count - 1ul
              : (right_width >= params.input_count - anchor
                     ? params.input_count - 1ul
                     : anchor + right_width - 1ul);
      uint value;
      if (left / block_width == right / block_width) {
        value = left % block_width == 0ul ? input[right] : output[left];
      } else {
        value = )MSL";
  source += combine;
  source += R"MSL((output[left], input[right]);
      }
      output[index] = )MSL";
  append_map_chain_expression(source, fusion, MapSlot::After, "value", "index");
  source += R"MSL(;
    }
)MSL";
  append_increment(source);
  source += "  }\n";
}

} // namespace

bool emit_metal_multipass(const RangeExec &execution,
                          const rund::kernel::ArtifactKey &key,
                          const DeviceVsmWindowFusion &fusion,
                          const DeviceVsmWindowMapSources &sources,
                          std::string &source) noexcept {
  try {
    source.clear();
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
    source += "kernel void ";
    source += entry_name(key);
    source += R"MSL((
    constant RangeParams& params [[buffer(0)]],
    device uint* input [[buffer(1)]],
    device uint* output [[buffer(2)]],
    constant RundDeviceVsmConfig& rund_vsm [[buffer(3)]],
    device atomic_uint* rund_vsm_result [[buffer(4)]],
    uint rund_local [[thread_index_in_threadgroup]],
    uint rund_worker [[threadgroup_position_in_grid]]) {
  if (rund_worker != 0u || rund_local != 0u) { return; }
)MSL";
    if (execution.candidate() == RangePath::PrefixDifference) {
      append_prefix(source, fusion);
    } else {
      append_extreme(source, execution, fusion);
    }
    source += "}\n";
    return !entry_name(key).empty();
  } catch (...) {
    source.clear();
    return false;
  }
}

} // namespace rund::node::accel::detail::device_vsm_window_source
