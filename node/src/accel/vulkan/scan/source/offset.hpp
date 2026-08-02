#pragma once

#include "prefix.hpp"

namespace rund::node::accel::detail {

template <typename Source>
inline void AppendVulkanScanOffset(Source &source,
                                   const rund::kernel::ScanElement element) {
  const char *const type = VulkanScanScalar(element);
  source += "void main() {\n";
  source += "  const uint lane = gl_LocalInvocationID.x;\n";
  source += "  const uint64_t block = scan_dispatch.base_block + "
            "uint64_t(gl_WorkGroupID.x);\n";
  source += "  if (block >= params.block_count) { return; }\n";
  source += "  const uint64_t block_size = params.block_size;\n";
  source += "  const uint64_t begin = block * block_size;\n";
  source += "  const uint64_t active_count = ActiveCount();\n";
  source += "  if (begin >= active_count) { return; }\n";
  source += "  const uint64_t end = min(begin + block_size, "
            "active_count);\n";
  source += "  const uint64_t lane_size = uint64_t(1) + "
            "(block_size - uint64_t(1)) / uint64_t(kScanWidth);\n";
  source += "  const uint64_t lane_begin = begin + uint64_t(lane) * "
            "lane_size;\n";
  source += "  const uint64_t lane_end = min(lane_begin + lane_size, end);\n";
  source += "  const ";
  source += type;
  source += " offset = totals[uint(block)];\n";
  source += "  uint bad = 0u;\n";
  source += "  for (uint64_t index = lane_begin; index < lane_end; "
            "++index) {\n";
  source += "    const ";
  source += type;
  source += " local = output_values[uint(index)];\n";
  source += "    const ";
  source += type;
  source += " value = input_values[uint(index)];\n";
  source += "    const ";
  source += type;
  source += " global = offset + local;\n";
  source += "    const ";
  source += type;
  source += " previous = params.inclusive != 0u ? global - value : global;\n";
  source += "    const ";
  source += type;
  source += " next = previous + value;\n";
  source += "    bad |= ScanOverflow(previous, value, next);\n";
  source += "    output_values[uint(index)] = global;\n";
  source += "  }\n";
  source += "  if (bad != 0u) { atomicOr(status[0], bad); }\n";
  source += "  if (lane == 0u) { const uint invalid = block == "
            "uint64_t(0) ? CountInvalid() : 0u; "
            "if (invalid != 0u) { atomicOr(status[0], invalid); "
            "} }\n";
  source += "}\n";
}

} // namespace rund::node::accel::detail
