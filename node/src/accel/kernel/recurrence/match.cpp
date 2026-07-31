#include "match.hpp"

#include "../../primitive/shape.hpp"

#include <cstring>

namespace rund::node::accel::detail {
using rund::kernel::BindingSet;
using rund::kernel::ComputeDispatchWindow;
using rund::kernel::ComputePlan;
using rund::kernel::LoweringArtifact;
using rund::kernel::ResidentBindingRange;
using rund::kernel::ResidentBufferRef;

namespace {

[[nodiscard]] bool SameRef(const ResidentBufferRef *const left,
                           const ResidentBufferRef *const right) noexcept {
  return left != nullptr && right != nullptr && left->id == right->id &&
         left->bytes == right->bytes &&
         left->offset_bytes == right->offset_bytes &&
         left->element_bytes == right->element_bytes &&
         left->stride_bytes == right->stride_bytes &&
         left->count == right->count;
}

[[nodiscard]] bool SameOwner(const ResidentBindingRange &left,
                             const std::uint64_t left_index,
                             const ResidentBindingRange &right,
                             const std::uint64_t right_index) noexcept {
  const std::shared_ptr<void> *const left_owner = left.handle(left_index);
  const std::shared_ptr<void> *const right_owner = right.handle(right_index);
  return left_owner != nullptr && right_owner != nullptr &&
         *left_owner != nullptr && *left_owner == *right_owner;
}

[[nodiscard]] bool SameParams(const BindingSet &left,
                              const BindingSet &right) noexcept {
  return left.param_bytes == right.param_bytes &&
         left.param_data_bytes == right.param_data_bytes &&
         left.param_data_bytes == left.param_bytes &&
         (left.param_data_bytes == 0u ||
          (left.param_data != nullptr && right.param_data != nullptr &&
           std::memcmp(left.param_data, right.param_data,
                       static_cast<std::size_t>(left.param_data_bytes)) == 0));
}

} // namespace

[[nodiscard]] bool SamePlan(const ComputePlan &left,
                            const ComputePlan &right) noexcept {
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

[[nodiscard]] bool SameView(const ResidentBindingRange &left,
                            const std::uint64_t left_index,
                            const ResidentBindingRange &right,
                            const std::uint64_t right_index) noexcept {
  return SameRef(left.ref(left_index), right.ref(right_index)) &&
         SameOwner(left, left_index, right, right_index);
}

[[nodiscard]] bool Aliases(const ResidentBindingRange &left,
                           const std::uint64_t left_index,
                           const ResidentBindingRange &right,
                           const std::uint64_t right_index) noexcept {
  const ResidentBufferRef *const left_ref = left.ref(left_index);
  const ResidentBufferRef *const right_ref = right.ref(right_index);
  const std::shared_ptr<void> *const left_owner = left.handle(left_index);
  const std::shared_ptr<void> *const right_owner = right.handle(right_index);
  if (left_ref == nullptr || right_ref == nullptr || left_owner == nullptr ||
      right_owner == nullptr || *left_owner == nullptr ||
      *right_owner == nullptr) {
    return true;
  }
  const bool same_owner = *left_owner == *right_owner;
  if (!same_owner && left_ref->id != right_ref->id) {
    return false;
  }
  return left_ref->id != right_ref->id ||
         ResidentOverlap(*left_ref, *right_ref);
}

[[nodiscard]] bool SameWindows(const BoundStep &left,
                               const BoundStep &right) noexcept {
  if (!left.map_windows.ok || !right.map_windows.ok ||
      left.map_windows.size() == 0u ||
      left.map_windows.size() != right.map_windows.size()) {
    return false;
  }
  for (std::uint64_t index = 0u; index < left.map_windows.size(); ++index) {
    const ComputeDispatchWindow &a = left.map_windows.data()[index];
    const ComputeDispatchWindow &b = right.map_windows.data()[index];
    if (a.begin_sequence != b.begin_sequence || a.tile_count != b.tile_count) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool SameMapBinding(const BindingSet &left,
                                  const BindingSet &right) noexcept {
  return left.tile_count == right.tile_count &&
         left.lane_count == right.lane_count &&
         left.logical_offset == right.logical_offset &&
         left.op_hash_hi == right.op_hash_hi &&
         left.op_hash_lo == right.op_hash_lo && left.api == right.api &&
         left.scalar == right.scalar && left.domain == right.domain &&
         left.input_bytes_per_tile == right.input_bytes_per_tile &&
         left.output_bytes_per_tile == right.output_bytes_per_tile &&
         left.metadata_bytes_per_tile == right.metadata_bytes_per_tile &&
         left.input_buffer_count == right.input_buffer_count &&
         left.output_buffer_count == right.output_buffer_count &&
         left.input_element_byte_count == right.input_element_byte_count &&
         left.output_element_byte_count == right.output_element_byte_count &&
         left.sequence_tiles == nullptr && right.sequence_tiles == nullptr &&
         left.sequence_tile_count == 0u && right.sequence_tile_count == 0u &&
         SameParams(left, right);
}

[[nodiscard]] bool SameArtifact(const LoweringArtifact &left,
                                const LoweringArtifact &right) noexcept {
  return left.ok && right.ok && left.key == right.key &&
         left.kind == right.kind && left.source_text == right.source_text &&
         left.canonical_ir_bytes == right.canonical_ir_bytes &&
         left.metadata.map.op_hash_hi == right.metadata.map.op_hash_hi &&
         left.metadata.map.op_hash_lo == right.metadata.map.op_hash_lo &&
         left.metadata.binding_accesses == right.metadata.binding_accesses &&
         left.metadata.binding_names == right.metadata.binding_names &&
         left.metadata.input_element_bytes ==
             right.metadata.input_element_bytes &&
         left.metadata.output_element_bytes ==
             right.metadata.output_element_bytes &&
         left.metadata.param_storage == right.metadata.param_storage &&
         left.metadata.read_count == right.metadata.read_count &&
         left.metadata.direct_read_mask ==
             right.metadata.direct_read_mask &&
         left.metadata.write_count == right.metadata.write_count &&
         left.metadata.uses_index == right.metadata.uses_index &&
         left.metadata.ok == right.metadata.ok;
}

[[nodiscard]] bool ExactRecurrenceMarker(
    const std::span<const BackendBatchEntry> entries) noexcept {
  if (entries.size() <= 1u ||
      entries.size() > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  const std::uint32_t logical = entries.front().recurrence.logical_step;
  const std::uint32_t bound = static_cast<std::uint32_t>(entries.size());
  for (std::size_t index = 0u; index < entries.size(); ++index) {
    const BackendRecurrence marker = entries[index].recurrence;
    if (marker.logical_step != logical || marker.bound != bound ||
        marker.iteration != index || marker.window != nullptr) {
      return false;
    }
  }
  return true;
}
} // namespace rund::node::accel::detail
