#pragma once

#include "../../../../../../backend/match.hpp"
#include "../../../../../../kernel/backend/execute.hpp"
#include "../../../../../runtime/map/resources.hpp"
#include "../../../residency/model.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] inline bool MetalSpatialWindowResidentRefPhysicalEqual(
    const rund::kernel::ResidentBufferRef &left,
    const rund::kernel::ResidentBufferRef &right) noexcept {
  return left.id == right.id && left.bytes == right.bytes &&
         left.offset_bytes == right.offset_bytes &&
         left.element_bytes == right.element_bytes &&
         left.stride_bytes == right.stride_bytes && left.count == right.count;
}

[[nodiscard]] inline bool MetalSpatialWindowResidentRefEqual(
    const rund::kernel::ResidentBufferRef &left,
    const rund::kernel::ResidentBufferRef &right) noexcept {
  return MetalSpatialWindowResidentRefPhysicalEqual(left, right) &&
         left.usage == right.usage;
}

[[nodiscard]] inline bool MetalSpatialWindowResidentRefValid(
    const rund::kernel::ResidentBufferRef &ref) noexcept {
  return ref.id != 0u && ref.bytes != 0u && ref.count != 0u &&
         ref.element_bytes != 0u && ref.stride_bytes >= ref.element_bytes &&
         ref.offset_bytes <= ref.bytes;
}

[[nodiscard]] inline std::uint64_t MetalSpatialWindowFixedFormatKey(
    const rund::kernel::ComputeFixedFormat format) noexcept {
  return static_cast<std::uint64_t>(format.integer_bits) |
         (static_cast<std::uint64_t>(format.fraction_bits) << 8u) |
         (static_cast<std::uint64_t>(format.rounding) << 16u) |
         (static_cast<std::uint64_t>(format.overflow) << 24u) |
         (static_cast<std::uint64_t>(format.approximation) << 32u);
}

[[nodiscard]] inline std::array<std::uint64_t, 11u>
MetalSpatialWindowDescriptorKey(const operation::Window &window) noexcept {
  return {static_cast<std::uint64_t>(window.desc.op),
          static_cast<std::uint64_t>(window.desc.element),
          static_cast<std::uint64_t>(window.desc.boundary),
          static_cast<std::uint64_t>(window.desc.domain),
          MetalSpatialWindowFixedFormatKey(window.desc.fixed_format),
          static_cast<std::uint64_t>(window.desc.count_source),
          window.desc.input_count, window.desc.output_count,
          window.desc.window_size, window.desc.stride, window.desc.pad_left};
}

[[nodiscard]] inline std::array<std::uint64_t, 17u>
MetalSpatialWindowPlanKey(const operation::Window &window) noexcept {
  return {static_cast<std::uint64_t>(window.plan.op),
          static_cast<std::uint64_t>(window.plan.element),
          static_cast<std::uint64_t>(window.plan.boundary),
          static_cast<std::uint64_t>(window.plan.domain),
          MetalSpatialWindowFixedFormatKey(window.plan.fixed_format),
          static_cast<std::uint64_t>(window.plan.count_source),
          window.plan.input_count, window.plan.output_count,
          window.plan.window_size, window.plan.stride, window.plan.pad_left,
          window.plan.element_bytes, window.plan.input_bytes,
          window.plan.output_bytes, window.plan.temp_bytes,
          window.plan.pass_count, static_cast<std::uint64_t>(window.plan.ok)};
}

[[nodiscard]] inline std::array<std::uint64_t, 12u>
MetalSpatialWindowRangeKey(const RangePlan &range) noexcept {
  const RangeShape &shape = range.shape();
  const std::optional<rund::kernel::u64> span = shape.affine_span();
  return {static_cast<std::uint64_t>(shape.boundary()), shape.input_count(),
          shape.output_count(), shape.window_size(), shape.stride(),
          shape.padding(), shape.element_bytes(), shape.payload_bytes(),
          shape.output_bytes(), shape.right_extent(),
          span.has_value() ? *span : 0u,
          static_cast<std::uint64_t>(shape.centered_clamp())};
}

[[nodiscard]] inline std::uint64_t
MetalSpatialWindowHaloFrame(const RangeCandidate &candidate) noexcept {
  return static_cast<std::uint64_t>(candidate.width()) +
         2u * static_cast<std::uint64_t>(candidate.radius_capacity());
}

[[nodiscard]] inline std::uint64_t
MetalSpatialWindowHaloPayload(const RangeCandidate &candidate,
                              const RangeShape &shape) noexcept {
  return MetalSpatialWindowHaloFrame(candidate) * shape.element_bytes();
}

[[nodiscard]] inline bool MetalSpatialWindowSameComputeMap(
    const rund::kernel::ComputeMap &left,
    const rund::kernel::ComputeMap &right) noexcept {
  return left.op_hash_hi == right.op_hash_hi &&
         left.op_hash_lo == right.op_hash_lo && left.api == right.api &&
         left.scalar == right.scalar && left.domain == right.domain &&
         left.fixed_format == right.fixed_format &&
         left.input_buffer_count == right.input_buffer_count &&
         left.output_buffer_count == right.output_buffer_count &&
         left.input_bytes_per_tile == right.input_bytes_per_tile &&
         left.output_bytes_per_tile == right.output_bytes_per_tile &&
         left.param_bytes == right.param_bytes &&
         left.metadata_bytes_per_tile == right.metadata_bytes_per_tile;
}

[[nodiscard]] inline bool MetalSpatialWindowSameResourceSummary(
    const rund::kernel::ComputeResourceSummary &left,
    const rund::kernel::ComputeResourceSummary &right) noexcept {
  return left.analysis_version == right.analysis_version &&
         left.peak_live_words == right.peak_live_words &&
         left.direct_read_count == right.direct_read_count &&
         left.uniform_read_count == right.uniform_read_count &&
         left.indexed_read_count == right.indexed_read_count &&
         left.write_count == right.write_count && left.ok == right.ok &&
         SameReason(left.reason, right.reason);
}

[[nodiscard]] inline bool MetalSpatialWindowSameMetadata(
    const rund::kernel::ExecutionMetadata &left,
    const rund::kernel::ExecutionMetadata &right) noexcept {
  return MetalSpatialWindowSameComputeMap(left.map, right.map) &&
         MetalSpatialWindowSameResourceSummary(left.resource_summary,
                                               right.resource_summary) &&
         left.param_storage == right.param_storage &&
         left.input_element_bytes == right.input_element_bytes &&
         left.output_element_bytes == right.output_element_bytes &&
         left.binding_accesses == right.binding_accesses &&
         left.binding_names == right.binding_names &&
         left.read_routes == right.read_routes &&
         left.direct_read_mask == right.direct_read_mask &&
         left.uniform_read_mask == right.uniform_read_mask &&
         left.read_count == right.read_count &&
         left.write_count == right.write_count &&
         left.uses_index == right.uses_index && left.ok == right.ok &&
         SameReason(left.reason, right.reason);
}

[[nodiscard]] inline bool MetalSpatialWindowSameParsedBinding(
    const rund::kernel::compute_lowering_detail::ParsedBinding &left,
    const rund::kernel::compute_lowering_detail::ParsedBinding &right) noexcept {
  return left.kind == right.kind && left.numeric_mode == right.numeric_mode &&
         left.name == right.name && left.element_bytes == right.element_bytes &&
         left.floating_point_param == right.floating_point_param &&
         left.value_bytes == right.value_bytes;
}

[[nodiscard]] inline bool MetalSpatialWindowSameParsedNode(
    const rund::kernel::compute_lowering_detail::ParsedNode &left,
    const rund::kernel::compute_lowering_detail::ParsedNode &right) noexcept {
  return left.op == right.op && left.lhs == right.lhs &&
         left.rhs == right.rhs && left.aux == right.aux &&
         left.fixed_format == right.fixed_format;
}

#endif

} // namespace rund::node::accel::detail
