#pragma once

#include "block.hpp"

namespace rund::node::accel::detail {

inline void AppendVulkanScanPrefix(std::string &source,
                                   const rund::kernel::ScanElement element) {
  const char *const type = VulkanScanScalar(element);
  const char *const zero = VulkanScanZero(element);
  source += "shared ";
  source += type;
  source += " chunk_totals[2][kScanWidth];\n";
  source += "void main() {\n";
  source += "  const uint lane = gl_LocalInvocationID.x;\n";
  source += "  const uint64_t block_count = params.block_count;\n";
  source += "  const uint64_t chunk_size = uint64_t(1) + "
            "(block_count - uint64_t(1)) / uint64_t(kScanWidth);\n";
  source += "  const uint64_t begin = min(uint64_t(lane) * chunk_size, "
            "block_count);\n";
  source += "  const uint64_t end = min(begin + chunk_size, block_count);\n";
  source += "  ";
  source += type;
  source += " running = ";
  source += zero;
  source += ";\n";
  source += "  for (uint64_t block = begin; block < end; ++block) {\n";
  source += "    const ";
  source += type;
  source += " value = totals[uint(block)];\n";
  source += "    totals[uint(block)] = running;\n";
  source += "    running += value;\n";
  source += "  }\n";
  source += "  chunk_totals[0][lane] = running;\n";
  source += "  barrier();\n";
  source += "  uint bank = 0u;\n";
  source += "  for (uint step = 1u; step < kScanWidth; step <<= 1u) {\n";
  source += "    const uint next_bank = bank ^ 1u;\n";
  source += "    const ";
  source += type;
  source += " addend = lane >= step ? chunk_totals[bank][lane - step] : ";
  source += zero;
  source += ";\n";
  source += "    chunk_totals[next_bank][lane] = "
            "chunk_totals[bank][lane] + addend;\n";
  source += "    barrier();\n";
  source += "    bank = next_bank;\n";
  source += "  }\n";
  source += "  const ";
  source += type;
  source += " offset = lane == 0u ? ";
  source += zero;
  source += " : chunk_totals[bank][lane - 1u];\n";
  source += "  for (uint64_t block = begin; block < end; ++block) {\n";
  source += "    totals[uint(block)] += offset;\n";
  source += "  }\n";
  source += "}\n";
}

} // namespace rund::node::accel::detail
