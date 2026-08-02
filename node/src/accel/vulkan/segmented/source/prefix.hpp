#pragma once

#include "prelude.hpp"

namespace rund::node::accel::detail {

template <typename Source>
inline void
AppendSegmentedPrefix(Source &source,
                      const rund::kernel::SegmentedScanElement element) {
  const char *const type = VulkanSegmentedType(element);
  const char *const zero =
      element == rund::kernel::SegmentedScanElement::U64 ? "uint64_t(0)" : "0u";
  source += "  const uint lane = gl_LocalInvocationID.x;\n";
  source += "  const uint64_t chunk = uint64_t(1) + "
            "(params.block_count - uint64_t(1)) / "
            "uint64_t(kSegmentedWidth);\n";
  source += "  const uint64_t begin = min(uint64_t(lane) * chunk, "
            "params.block_count);\n";
  source += "  const uint64_t end = min(begin + chunk, "
            "params.block_count);\n";
  source += "  if (lane == 0u) { segment_status = 0u; }\n";
  source += "  barrier();\n";
  source += "  ";
  source += type;
  source += " summary = ";
  source += zero;
  source += "; uint has_head = 0u; uint bad = 0u;\n";
  source += "  for (uint64_t block = begin; block < end; ++block) {\n";
  source += "    const uint64_t block_begin = block * params.block_size;\n";
  source += "    const uint64_t block_end = min(block_begin + "
            "params.block_size, "
            "params.element_count);\n";
  source += "    const uint len = uint(block_end - block_begin);\n";
  source += "    const uint first = first_heads[uint(block)];\n";
  source += "    const ";
  source += type;
  source += " tail = offsets[uint(block)];\n";
  source += "    bad = max(bad, status[uint(block)]);\n";
  source += "    const uint reset = first < len ? 1u : 0u;\n";
  source += "    summary = reset != 0u ? tail : summary + tail;\n";
  source += "    has_head |= reset;\n";
  source += "  }\n";
  source += "  segment_values[0][lane] = summary;\n";
  source += "  segment_flags[0][lane] = has_head;\n";
  source += "  if (bad != 0u) { atomicMax(segment_status, bad); }\n";
  source += "  barrier();\n";
  source += "  uint bank = 0u;\n";
  source += "  for (uint step = 1u; step < kSegmentedWidth; "
            "step <<= 1u) {\n";
  source += "    const uint next_bank = bank ^ 1u;\n";
  source += "    const ";
  source += type;
  source += " left = lane >= step ? segment_values[bank][lane - step] : ";
  source += zero;
  source += ";\n";
  source += "    const uint left_head = lane >= step ? "
            "segment_flags[bank][lane - step] : 0u;\n";
  source += "    ";
  source += type;
  source += " value = segment_values[bank][lane];\n";
  source += "    uint reset = segment_flags[bank][lane];\n";
  source += "    if (lane >= step) { if (reset == 0u) { value += left; } "
            "reset |= left_head; }\n";
  source += "    segment_values[next_bank][lane] = value; "
            "segment_flags[next_bank][lane] = reset;\n";
  source += "    barrier();\n";
  source += "    bank = next_bank;\n";
  source += "  }\n";
  source += "  ";
  source += type;
  source += " carry = lane == 0u ? ";
  source += zero;
  source += " : segment_values[bank][lane - 1u];\n";
  source += "  for (uint64_t block = begin; block < end; ++block) {\n";
  source += "    const uint64_t block_begin = block * params.block_size;\n";
  source += "    const uint64_t block_end = min(block_begin + "
            "params.block_size, params.element_count);\n";
  source += "    const uint len = uint(block_end - block_begin);\n";
  source += "    const uint first = first_heads[uint(block)];\n";
  source += "    const ";
  source += type;
  source += " tail = offsets[uint(block)];\n";
  source += "    offsets[uint(block)] = carry;\n";
  source += "    carry = first < len ? tail : carry + tail;\n";
  source += "  }\n";
  source += "  barrier();\n";
  source += "  if (lane == 0u) { status[0] = segment_status; }\n";
}

} // namespace rund::node::accel::detail
