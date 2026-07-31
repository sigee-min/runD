#pragma once

#include "prelude.hpp"

namespace rund::node::accel::detail {

inline void
AppendSegmentedBlock(std::string &source,
                     const rund::kernel::SegmentedScanElement element) {
  const char *const type = VulkanSegmentedType(element);
  const char *const zero =
      element == rund::kernel::SegmentedScanElement::U64 ? "uint64_t(0)" : "0u";
  source += "  const uint64_t block = segmented_dispatch.base_block + "
            "uint64_t(gl_WorkGroupID.x);\n";
  source += "  if (block >= params.block_count) { return; }\n";
  source += "  const uint lane = gl_LocalInvocationID.x;\n";
  source += "  const uint64_t begin = block * params.block_size;\n";
  source += "  const uint64_t end = min(begin + params.block_size, "
            "params.element_count);\n";
  source += "  const uint len = uint(end - begin);\n";
  source += "  if (lane == 0u) { segment_carry = ";
  source += zero;
  source += "; segment_seen = 0u; segment_first = len; "
            "segment_status = 0u; }\n";
  source += "  barrier();\n";
  source += "  for (uint tile = 0u; tile < len; tile += "
            "kSegmentedWidth) {\n";
  source += "    const uint local = tile + lane;\n";
  source += "    const bool valid = local < len;\n";
  source += "    const uint index = uint(begin) + local;\n";
  source += "    const uint head = valid ? heads[index] : 0u;\n";
  source += "    uint bad = valid && (head > 1u || "
            "(index == 0u && head != 1u)) ? 2u : 0u;\n";
  source += "    if (valid && head == 1u) { atomicMin(segment_first, "
            "local); }\n";
  source += "    segment_values[0][lane] = valid ? input_values[index] : ";
  source += zero;
  source += ";\n";
  source += "    segment_flags[0][lane] = valid ? min(head, 1u) : 0u;\n";
  source += "    barrier();\n";
  source += "    uint bank = 0u;\n";
  source += "    for (uint step = 1u; step < kSegmentedWidth; "
            "step <<= 1u) {\n";
  source += "      const uint next_bank = bank ^ 1u;\n";
  source += "      const ";
  source += type;
  source += " left = lane >= step ? segment_values[bank][lane - step] : ";
  source += zero;
  source += ";\n";
  source += "      const uint left_head = lane >= step ? "
            "segment_flags[bank][lane - step] : 0u;\n";
  source += "      ";
  source += type;
  source += " value = segment_values[bank][lane];\n";
  source += "      uint has_head = segment_flags[bank][lane];\n";
  source += "      if (lane >= step) { if (has_head == 0u) { value += "
            "left; } has_head |= left_head; }\n";
  source += "      segment_values[next_bank][lane] = value; "
            "segment_flags[next_bank][lane] = has_head;\n";
  source += "      barrier();\n";
  source += "      bank = next_bank;\n";
  source += "    }\n";
  source += "    const ";
  source += type;
  source += " local_next = segment_values[bank][lane];\n";
  source += "    const ";
  source += type;
  source += " next = segment_flags[bank][lane] == 0u ? segment_carry + "
            "local_next : local_next;\n";
  source += "    const ";
  source += type;
  source += " previous = head == 1u ? ";
  source += zero;
  source += " : next - (valid ? input_values[index] : ";
  source += zero;
  source += ");\n";
  source += "    const bool owns = block == uint64_t(0) || "
            "segment_seen != 0u || segment_flags[bank][lane] != 0u;\n";
  source += "    if (valid && owns && "
            "SegmentedOverflow(previous, input_values[index], next) != 0u) "
            "{ bad = max(bad, 1u); }\n";
  source += "    if (valid) { output_values[index] = "
            "params.inclusive != 0u ? next : previous; }\n";
  source += "    if (bad != 0u) { atomicMax(segment_status, bad); }\n";
  source += "    barrier();\n";
  source += "    if (lane == 0u) {\n";
  source += "      const uint tile_count = min(kSegmentedWidth, len - tile);\n";
  source += "      const uint last = tile_count - 1u;\n";
  source += "      segment_carry = segment_flags[bank][last] == 0u ? "
            "segment_carry + segment_values[bank][last] : "
            "segment_values[bank][last];\n";
  source += "      segment_seen |= segment_flags[bank][last];\n";
  source += "    }\n";
  source += "    barrier();\n";
  source += "  }\n";
  source += "  if (lane == 0u) { offsets[uint(block)] = segment_carry; "
            "first_heads[uint(block)] = segment_first; "
            "status[uint(block)] = segment_status; }\n";
}

} // namespace rund::node::accel::detail
