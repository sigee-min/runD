#pragma once

#include "prelude.hpp"

namespace rund::node::accel::detail {

inline void AppendVulkanScanBlock(std::string &source,
                                  const rund::kernel::ScanElement element,
                                  const bool inclusive) {
  const char *const type = VulkanScanScalar(element);
  const char *const zero = VulkanScanZero(element);
  source += "shared ";
  source += type;
  source += " lane_totals[kScanWidth];\n";
  source += "void main() {\n";
  source += "  const uint lane = gl_LocalInvocationID.x;\n";
  source += "  const uint64_t block = scan_dispatch.base_block + "
            "uint64_t(gl_WorkGroupID.x);\n";
  source += "  if (block >= params.block_count) { return; }\n";
  source += "  const uint64_t block_size = params.block_size;\n";
  source += "  const uint64_t begin = block * block_size;\n";
  source += "  const uint64_t active_count = ActiveCount();\n";
  source += "  if (begin >= active_count) {\n";
  source += "    if (lane == 0u) { totals[uint(block)] = ";
  source += zero;
  source += "; const uint invalid = block == uint64_t(0) ? "
            "CountInvalid() : 0u; if (invalid != 0u) { "
            "atomicOr(status[0], invalid); } }\n";
  source += "    return;\n";
  source += "  }\n";
  source += "  const uint64_t end = min(begin + block_size, "
            "active_count);\n";
  source += "  const uint64_t lane_size = uint64_t(1) + "
            "(block_size - uint64_t(1)) / uint64_t(kScanWidth);\n";
  source += "  const uint64_t lane_begin = begin + uint64_t(lane) * "
            "lane_size;\n";
  source += "  const uint64_t lane_end = min(lane_begin + lane_size, end);\n";
  source += "  ";
  source += type;
  source += " running = ";
  source += zero;
  source += ";\n";
  source += "  for (uint64_t index = lane_begin; index < lane_end; "
            "++index) { running += input_values[uint(index)]; }\n";
  source += "  lane_totals[lane] = running;\n";
  source += "  barrier();\n";
  source += "  for (uint step = 1u; step < kScanWidth; step <<= 1u) {\n";
  source += "    const ";
  source += type;
  source += " addend = lane >= step ? lane_totals[lane - step] : ";
  source += zero;
  source += ";\n";
  source += "    barrier();\n";
  source += "    lane_totals[lane] += addend;\n";
  source += "    barrier();\n";
  source += "  }\n";
  source += "  running = lane == 0u ? ";
  source += zero;
  source += " : lane_totals[lane - 1u];\n";
  source += "  uint bad = 0u;\n";
  source += "  for (uint64_t index = lane_begin; index < lane_end; "
            "++index) {\n";
  source += "    const ";
  source += type;
  source += " value = input_values[uint(index)];\n";
  source += "    const ";
  source += type;
  source += " next = running + value;\n";
  source += "    if (params.block_count == uint64_t(1)) { bad |= "
            "ScanOverflow(running, value, next); }\n";
  source += "    output_values[uint(index)] = ";
  source += inclusive ? "next" : "running";
  source += ";\n";
  source += "    running = next;\n";
  source += "  }\n";
  source += "  if (bad != 0u) { atomicOr(status[0], bad); }\n";
  source += "  if (lane == 0u) {\n";
  source += "    totals[uint(block)] = lane_totals[kScanWidth - 1u];\n";
  source += "    const uint invalid = block == uint64_t(0) ? "
            "CountInvalid() : 0u;\n";
  source += "    if (invalid != 0u) { atomicOr(status[0], "
            "invalid); }\n";
  source += "  }\n";
  source += "}\n";
}

} // namespace rund::node::accel::detail
