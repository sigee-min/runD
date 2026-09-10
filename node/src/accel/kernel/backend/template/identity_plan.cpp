#include "identity_internal.hpp"

#include <cstddef>

namespace rund::node::accel::detail::backend_template_plan {

bool same_plan(const rund::kernel::ComputePlan &left,
               const rund::kernel::ComputePlan &right) noexcept {
  return left.phase_id == right.phase_id &&
         left.tile_count == right.tile_count &&
         left.op_hash_hi == right.op_hash_hi &&
         left.op_hash_lo == right.op_hash_lo && left.api == right.api &&
         left.scalar == right.scalar && left.domain == right.domain &&
         left.fixed_format == right.fixed_format &&
         left.input_buffer_count == right.input_buffer_count &&
         left.output_buffer_count == right.output_buffer_count &&
         left.input_bytes_per_tile == right.input_bytes_per_tile &&
         left.output_bytes_per_tile == right.output_bytes_per_tile &&
         left.param_bytes == right.param_bytes &&
         left.metadata_bytes_per_tile == right.metadata_bytes_per_tile &&
         left.bytes_per_tile == right.bytes_per_tile &&
         left.staging_bytes == right.staging_bytes &&
         left.dispatch_window_tiles == right.dispatch_window_tiles &&
         left.dispatch_count == right.dispatch_count &&
         left.fixed_authoritative == right.fixed_authoritative &&
         left.ok == right.ok;
}

bool same_windows(const DispatchWindowStorage &left,
                  const DispatchWindowStorage &right) noexcept {
  if (left.ok != right.ok || left.size() != right.size()) {
    return false;
  }
  const auto *const left_data = left.data();
  const auto *const right_data = right.data();
  for (std::uint64_t index = 0u; index < left.size(); ++index) {
    if (left_data == nullptr || right_data == nullptr ||
        left_data[index].begin_sequence != right_data[index].begin_sequence ||
        left_data[index].tile_count != right_data[index].tile_count) {
      return false;
    }
  }
  return true;
}

bool same_layout(const KernelViewLayout *const left,
                 const KernelViewLayout *const right) noexcept {
  if (left == nullptr || right == nullptr) {
    return left == right;
  }
  if (left->size() != right->size()) {
    return false;
  }
  for (std::size_t index = 0u; index < left->size(); ++index) {
    const KernelViewSlot &a = (*left)[index];
    const KernelViewSlot &b = (*right)[index];
    if (a.binding != b.binding || a.slot != b.slot ||
        a.backing_bytes != b.backing_bytes ||
        a.offset_bytes != b.offset_bytes || a.count != b.count ||
        a.stride_bytes != b.stride_bytes ||
        a.element_bytes != b.element_bytes || a.usage != b.usage) {
      return false;
    }
  }
  return true;
}

bool same_layout(const KernelScratchLayout *const left,
                 const KernelScratchLayout *const right) noexcept {
  if (left == nullptr || right == nullptr) {
    return left == right;
  }
  if (left->size() != right->size()) {
    return false;
  }
  for (std::size_t index = 0u; index < left->size(); ++index) {
    const KernelScratchPage &a = (*left)[index];
    const KernelScratchPage &b = (*right)[index];
    if (a.slot != b.slot || a.bytes != b.bytes) {
      return false;
    }
  }
  return true;
}

} // namespace rund::node::accel::detail::backend_template_plan
