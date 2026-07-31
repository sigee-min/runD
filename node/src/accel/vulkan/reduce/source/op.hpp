#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] inline const char *
VulkanReduceInit(const rund::kernel::ReduceOp op, const bool u64,
                 const bool signed_domain) noexcept {
  if (op == rund::kernel::ReduceOp::CountNonzero && u64) {
    return "  sums[tid] = index < active_count ? (params.initial_pass != "
           "0u ? (input_values[source_index] != uint64_t(0) ? uint64_t(1) : "
           "uint64_t(0)) : input_values[source_index]) : uint64_t(0);\n";
  }
  if (op == rund::kernel::ReduceOp::CountNonzero) {
    return "  sums[tid] = index < active_count ? (params.initial_pass != "
           "0u ? (input_values[source_index] != 0u ? uint64_t(1) : "
           "uint64_t(0)) : uint64_t(input_values[source_index])) : "
           "uint64_t(0);\n";
  }
  if (op == rund::kernel::ReduceOp::Min && u64) {
    return signed_domain ? "  sums[tid] = index < active_count ? "
                           "input_values[source_index] : "
                           "uint64_t(0x7ffffffffffffffful);\n"
                         : "  sums[tid] = index < active_count ? "
                           "input_values[source_index] : "
                           "uint64_t(0xfffffffffffffffful);\n";
  }
  if (op == rund::kernel::ReduceOp::Min) {
    return signed_domain ? "  sums[tid] = index < active_count ? "
                           "uint64_t(int64_t(int(input_values[source_index]))) "
                           ": uint64_t(0x7fffffffu);\n"
                         : "  sums[tid] = index < active_count ? "
                           "uint64_t(input_values[source_index]) : "
                           "uint64_t(0xffffffffu);\n";
  }
  if (op == rund::kernel::ReduceOp::Max && signed_domain && u64) {
    return "  sums[tid] = index < active_count ? "
           "input_values[source_index] : uint64_t(0x8000000000000000ul);\n";
  }
  if (op == rund::kernel::ReduceOp::Max && signed_domain) {
    return "  sums[tid] = index < active_count ? "
           "uint64_t(int64_t(int(input_values[source_index]))) : "
           "uint64_t(0xffffffff80000000ul);\n";
  }
  if (op == rund::kernel::ReduceOp::Sum && signed_domain && !u64) {
    return "  sums[tid] = index < active_count ? "
           "uint64_t(int64_t(int(input_values[source_index]))) : "
           "uint64_t(0);\n";
  }
  return "  sums[tid] = index < active_count ? "
         "uint64_t(input_values[source_index]) : uint64_t(0);\n";
}

[[nodiscard]] inline const char *
VulkanReduceCombine(const rund::kernel::ReduceOp op, const bool u64,
                    const bool signed_domain) noexcept {
  if (op == rund::kernel::ReduceOp::Min) {
    if (signed_domain && u64) {
      return "      const uint64_t combined = uint64_t(min(int64_t(sums[tid]), "
             "int64_t(rhs)));\n";
    }
    if (signed_domain) {
      return "      const uint64_t combined = "
             "uint64_t(min(int64_t(int(uint(sums[tid]))), "
             "int64_t(int(uint(rhs)))));\n";
    }
    return "      const uint64_t combined = min(sums[tid], rhs);\n";
  }
  if (op == rund::kernel::ReduceOp::Max) {
    if (signed_domain && u64) {
      return "      const uint64_t combined = uint64_t(max(int64_t(sums[tid]), "
             "int64_t(rhs)));\n";
    }
    if (signed_domain) {
      return "      const uint64_t combined = "
             "uint64_t(max(int64_t(int(uint(sums[tid]))), "
             "int64_t(int(uint(rhs)))));\n";
    }
    return "      const uint64_t combined = max(sums[tid], rhs);\n";
  }
  return "      const uint64_t combined = sums[tid] + rhs;\n";
}

[[nodiscard]] inline const char *
VulkanReduceOverflow(const rund::kernel::ReduceOp op, const bool u64,
                     const bool signed_domain) noexcept {
  if (op == rund::kernel::ReduceOp::Min || op == rund::kernel::ReduceOp::Max) {
    return "";
  }
  if (signed_domain && !u64) {
    return "      if (int64_t(combined) < int64_t(-2147483647 - 1) || "
           "int64_t(combined) > int64_t(2147483647)) { overflows[tid] = 1u; "
           "}\n";
  }
  if (signed_domain && op == rund::kernel::ReduceOp::CountNonzero) {
    return "      if ((combined & uint64_t(0x8000000000000000ul)) != "
           "uint64_t(0)) { overflows[tid] = 1u; }\n";
  }
  if (signed_domain) {
    return "      if ((((sums[tid] ^ rhs) & uint64_t(0x8000000000000000ul)) == "
           "uint64_t(0)) && (((sums[tid] ^ combined) & "
           "uint64_t(0x8000000000000000ul)) != uint64_t(0))) { overflows[tid] "
           "= 1u; }\n";
  }
  return u64 ? "      if (combined < sums[tid]) { overflows[tid] = 1u; }\n"
             : "      if (combined < sums[tid] || combined > "
               "uint64_t(0xffffffffu)) { overflows[tid] = 1u; }\n";
}

} // namespace rund::node::accel::detail
