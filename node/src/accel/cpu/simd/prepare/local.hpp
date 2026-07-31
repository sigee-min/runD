#pragma once

#include <cstddef>
#include <limits>

namespace rund::node::accel::cpu_simd_detail {
namespace {

using rund::kernel::ComputeScalar;
using rund::kernel::IrOp;
using rund::kernel::u32;
using rund::kernel::u64;
using rund::kernel::compute_lowering_detail::kParamBindingKind;
using rund::kernel::compute_lowering_detail::kReadBindingKind;
using rund::kernel::compute_lowering_detail::kWriteBindingKind;
using rund::kernel::compute_lowering_detail::ParsedBinding;
using rund::kernel::compute_lowering_detail::ParsedIR;
using rund::kernel::compute_lowering_detail::ParsedNode;
using rund::kernel::compute_lowering_detail::ScalarBytes;

[[nodiscard]] PreparedRun RejectPrepared(const char *const reason) {
  PreparedRun prepared;
  prepared.reason = reason;
  return prepared;
}

[[nodiscard]] bool FitsSize(const u64 value) noexcept {
  return value <= static_cast<u64>(std::numeric_limits<std::size_t>::max());
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
