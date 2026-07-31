#pragma once

#include "prelude.hpp"

namespace rund::node::accel::detail {

inline void
AppendSegmentedOffset(std::string &source,
                      const rund::kernel::SegmentedScanElement element) {
  const char *const type = VulkanSegmentedType(element);
  source += "  const uint64_t block = segmented_dispatch.base_block + "
            "uint64_t(gl_WorkGroupID.x);\n";
  source += "  if (block == uint64_t(0) || block >= params.block_count) { "
            "return; }\n";
  source += "  const uint lane = gl_LocalInvocationID.x;\n";
  source += "  const uint64_t begin = block * params.block_size;\n";
  source += "  const uint64_t end = min(begin + params.block_size, "
            "params.element_count);\n";
  source += "  const uint stop = min(first_heads[uint(block)], "
            "uint(end - begin));\n";
  source += "  ";
  source += type;
  source += " carry = offsets[uint(block)]; uint bad = 0u;\n";
  source += "  for (uint local = lane; local < stop; local += "
            "kSegmentedWidth) {\n";
  source += "    const uint index = uint(begin) + local;\n";
  source += "    const ";
  source += type;
  source += " value = input_values[index];\n";
  source += "    const ";
  source += type;
  source += " local_result = output_values[index];\n";
  source += "    const ";
  source += type;
  source += " local_previous = params.inclusive != 0u ? "
            "local_result - value : local_result;\n";
  source += "    const ";
  source += type;
  source += " previous = carry + local_previous;\n";
  source += "    const ";
  source += type;
  source += " next = previous + value;\n";
  source += "    bad = max(bad, SegmentedOverflow(previous, value, next));\n";
  source += "    output_values[index] = params.inclusive != 0u ? next : "
            "previous;\n";
  source += "  }\n";
  source += "  if (bad != 0u) { atomicMax(status[0], bad); }\n";
}

} // namespace rund::node::accel::detail
