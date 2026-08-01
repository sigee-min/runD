#include "admit.hpp"

#include "../../../buffer/resident/find.hpp"
#include "../../../resident.hpp"
#include "../../../resident/access.hpp"

#include <array>
#include <limits>
#include <mutex>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] bool U32Read(const NestedAggregateRead &read) noexcept {
  const rund::kernel::ResidentBufferRef &ref = read.read.source;
  return read.read.handle != nullptr &&
         read.element_bytes == sizeof(uint32_t) &&
         ref.element_bytes == sizeof(uint32_t) &&
         read.logical_count == ref.count && ref.count != 0u &&
         ref.stride_bytes >= sizeof(uint32_t) &&
         (ref.offset_bytes % sizeof(uint32_t)) == 0u &&
         (ref.stride_bytes % sizeof(uint32_t)) == 0u;
}

[[nodiscard]] bool U32Scalar(const BackendRead &read) noexcept {
  const rund::kernel::ResidentBufferRef &ref = read.source;
  return read.handle != nullptr && ref.count == 1u &&
         ref.element_bytes == sizeof(uint32_t) &&
         ref.stride_bytes >= sizeof(uint32_t) &&
         (ref.offset_bytes % sizeof(uint32_t)) == 0u &&
         (ref.stride_bytes % sizeof(uint32_t)) == 0u;
}

[[nodiscard]] bool U32Scalar(const rund::kernel::ResidentBufferRef &ref,
                             const std::shared_ptr<void> &handle) noexcept {
  return handle != nullptr && ref.count == 1u &&
         ref.element_bytes == sizeof(uint32_t) &&
         ref.stride_bytes >= sizeof(uint32_t) &&
         (ref.offset_bytes % sizeof(uint32_t)) == 0u &&
         (ref.stride_bytes % sizeof(uint32_t)) == 0u;
}

[[nodiscard]] bool U32Workspace(const NestedAggregateWorkspace &workspace,
                                const std::uint64_t count) noexcept {
  const rund::kernel::ResidentBufferRef &ref = workspace.ref;
  if (workspace.handle == nullptr || count == 0u || ref.id == 0u ||
      ref.usage != rund::kernel::kResidentUsageWrite ||
      ref.element_bytes != sizeof(std::uint32_t) ||
      ref.stride_bytes != sizeof(std::uint32_t) || ref.count < count ||
      (ref.offset_bytes % sizeof(std::uint32_t)) != 0u ||
      ref.offset_bytes > ref.bytes) {
    return false;
  }
  const std::uint64_t available = ref.bytes - ref.offset_bytes;
  return ref.count <= available / sizeof(std::uint32_t);
}

[[nodiscard]] bool DisjointWorkspace(const NestedAggregateWorkspace &left,
                                     const NestedAggregateWorkspace &right,
                                     const std::uint64_t count) noexcept {
  const bool same_id = left.ref.id == right.ref.id;
  const bool same_owner = left.handle == right.handle;
  if (same_id != same_owner) {
    return false;
  }
  if (!same_id) {
    return true;
  }
  if (count >
      std::numeric_limits<std::uint64_t>::max() / sizeof(std::uint32_t)) {
    return false;
  }
  const std::uint64_t left_bytes = count * sizeof(std::uint32_t);
  const std::uint64_t right_bytes = left_bytes;
  return (left.ref.offset_bytes <=
              std::numeric_limits<std::uint64_t>::max() - left_bytes &&
          left.ref.offset_bytes + left_bytes <= right.ref.offset_bytes) ||
         (right.ref.offset_bytes <=
              std::numeric_limits<std::uint64_t>::max() - right_bytes &&
          right.ref.offset_bytes + right_bytes <= left.ref.offset_bytes);
}

[[nodiscard]] bool SameRead(const BackendRead &left,
                            const BackendRead &right) noexcept {
  return left.handle == right.handle && left.source.id == right.source.id &&
         left.source.offset_bytes == right.source.offset_bytes &&
         left.source.element_bytes == right.source.element_bytes &&
         left.source.stride_bytes == right.source.stride_bytes &&
         left.source.count == right.source.count;
}

[[nodiscard]] bool ScalarExpr(const NestedScalarExpr &source,
                              MetalNestedScalarExpr &out) noexcept {
  if (source.op != NestedScalarOp::AddWrapU32) {
    return false;
  }
  const auto value = [](const NestedScalarValue source, std::uint32_t &target) {
    switch (source) {
    case NestedScalarValue::TileState:
      target = static_cast<std::uint32_t>(MetalNestedScalarValue::TileState);
      return true;
    case NestedScalarValue::TileCount:
      target = static_cast<std::uint32_t>(MetalNestedScalarValue::TileCount);
      return true;
    case NestedScalarValue::OuterState:
      target = static_cast<std::uint32_t>(MetalNestedScalarValue::OuterState);
      return true;
    case NestedScalarValue::Immediate:
      target = static_cast<std::uint32_t>(MetalNestedScalarValue::Immediate);
      return true;
    case NestedScalarValue::None:
      return false;
    }
    return false;
  };
  out = MetalNestedScalarExpr{.immediate = source.immediate};
  return value(source.lhs, out.lhs) && value(source.rhs, out.rhs);
}

} // namespace

rund::AccelCheck
AdmitMetalNestedAggregate(const NestedAggregate &source,
                          const PreparedPipelineStatusLayout &status,
                          const bool profile_steps, MetalKernelContext &context,
                          MetalNestedAggregate &out) {
  out = {};
  MetalNestedScalarExpr action{};
  MetalNestedScalarExpr fold{};
  const BackendRead &seed = source.publication.sources[0u];
  const rund::kernel::ResidentBufferRef &target = source.publication.target;
  const BackendWindow &window = source.window;
  if (!source.ready() || window.maximum == 0u || window.tile == 0u ||
      window.tile == std::numeric_limits<std::uint32_t>::max() ||
      window.tile > window.maximum || window.outer_bound == 0u ||
      window.inner_bound == 0u || source.seed.count == 0u) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }
  const std::uint64_t expected_outer =
      (static_cast<std::uint64_t>(window.maximum) + window.tile - 1u) /
      window.tile;
  const std::uint64_t seed_end = source.seed.end();
  bool declared_seed_range = source.seed.count == window.outer_bound &&
                             seed_end <= status.active_step_count;
  if (declared_seed_range) {
    const std::uint32_t first = status.declared_steps[source.seed.first];
    for (std::uint32_t outer = 0u; outer < source.seed.count; ++outer) {
      if (first > std::numeric_limits<std::uint32_t>::max() - outer ||
          status.declared_steps[source.seed.first + outer] != first + outer) {
        declared_seed_range = false;
        break;
      }
    }
  }
  bool profile_layout = !profile_steps;
  if (profile_steps && source.seed.first == 0u &&
      source.fold.end() == status.active_step_count &&
      status.active_step_count == status.declared_step_count) {
    profile_layout = true;
    for (std::uint32_t index = 0u; index < status.active_step_count; ++index) {
      if (status.declared_steps[index] != index) {
        profile_layout = false;
        break;
      }
    }
  }
  const std::uint64_t authored_actions =
      static_cast<std::uint64_t>(window.outer_bound) * window.inner_bound;
  if (source.kind != NestedAggregateKind::WindowIndexedReduceSumU32 ||
      window.has_terminal || window.outer_bound != expected_outer ||
      window.state != source.publication.state ||
      source.publication_index == NoNode || source.publication.final > 2u ||
      source.queue.logical_count < window.maximum || !U32Read(source.queue) ||
      !U32Read(source.domain) || !U32Read(source.count) ||
      source.count.logical_count != 1u ||
      !U32Workspace(source.tile_low, window.outer_bound) ||
      !U32Workspace(source.tile_status, window.outer_bound) ||
      !DisjointWorkspace(source.tile_low, source.tile_status,
                         window.outer_bound) ||
      !SameRead(source.count.read, window.count) || !U32Scalar(seed) ||
      !U32Scalar(target, source.publication.target_handle) ||
      source.failure.logical_step == NoNode || !declared_seed_range ||
      source.failure.logical_step != status.declared_steps[source.seed.first] ||
      !profile_layout || !source.profile.aggregate_profile_supported ||
      source.profile.authored_seed_occurrences != window.outer_bound ||
      source.profile.authored_action_occurrences != authored_actions ||
      source.profile.authored_fold_occurrences != window.outer_bound ||
      source.profile.seed_dispatches_per_occurrence == 0u ||
      source.profile.action_dispatches_per_occurrence == 0u ||
      source.profile.fold_dispatches_per_occurrence == 0u ||
      status.generation_stride == 0u ||
      source.failure.invalid_index_reason == 0u ||
      source.failure.reduce_overflow_reason == 0u ||
      source.failure.phase != rund::compute::PipelineNestedPhase::Seed ||
      !source.failure.inner_coordinate_unknown ||
      source.failure.count_overflow_reason == 0u ||
      context.adapter == nullptr ||
      source.action_expr.lhs != NestedScalarValue::TileState ||
      (source.action_expr.rhs != NestedScalarValue::TileCount &&
       source.action_expr.rhs != NestedScalarValue::Immediate) ||
      source.fold_expr.lhs != NestedScalarValue::OuterState ||
      source.fold_expr.rhs != NestedScalarValue::TileState ||
      !ScalarExpr(source.action_expr, action) ||
      !ScalarExpr(source.fold_expr, fold)) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }

  std::array<MetalResidentBufferResult, 7u> resolved{};
  MetalResidentState &resident = MetalResidents(*context.adapter);
  {
    std::lock_guard lock{resident.mutex};
    const auto resolve = [&](const BackendRead &read) {
      return ResolveMetalResidentBuffer(resident, read.source, read.handle,
                                        "accel_metal_resident_id_unavailable",
                                        true);
    };
    resolved[0u] = resolve(source.queue.read);
    resolved[1u] = resolve(source.domain.read);
    resolved[2u] = resolve(source.count.read);
    resolved[3u] = resolve(seed);
    resolved[4u] = ResolveMetalResidentBuffer(
        resident, target, source.publication.target_handle,
        "accel_metal_resident_id_unavailable", true);
    resolved[5u] = ResolveMetalResidentBuffer(
        resident, source.tile_low.ref, source.tile_low.handle,
        "accel_metal_resident_id_unavailable");
    resolved[6u] = ResolveMetalResidentBuffer(
        resident, source.tile_status.ref, source.tile_status.handle,
        "accel_metal_resident_id_unavailable");
  }
  for (const MetalResidentBufferResult &buffer : resolved) {
    if (!buffer.check.ok || buffer.device_buffer == nullptr) {
      return buffer.check.ok
                 ? rund::AccelCheck{false, "accel_metal_buffer_failed"}
                 : buffer.check;
    }
  }
  for (std::size_t index = 0u; index < resolved.size(); ++index) {
    out.buffers[index] = resolved[index].device_buffer;
  }
  out.params = MetalNestedAggregateParams{
      .queue_offset_words =
          source.queue.read.source.offset_bytes / sizeof(std::uint32_t),
      .queue_stride_words =
          source.queue.read.source.stride_bytes / sizeof(std::uint32_t),
      .domain_offset_words =
          source.domain.read.source.offset_bytes / sizeof(std::uint32_t),
      .domain_stride_words =
          source.domain.read.source.stride_bytes / sizeof(std::uint32_t),
      .count_offset_words =
          source.count.read.source.offset_bytes / sizeof(std::uint32_t),
      .seed_offset_words = seed.source.offset_bytes / sizeof(std::uint32_t),
      .target_offset_words = target.offset_bytes / sizeof(std::uint32_t),
      .tile_low_offset_words =
          source.tile_low.ref.offset_bytes / sizeof(std::uint32_t),
      .tile_status_offset_words =
          source.tile_status.ref.offset_bytes / sizeof(std::uint32_t),
      .queue_count = source.queue.logical_count,
      .domain_count = source.domain.logical_count,
      .maximum = window.maximum,
      .tile = window.tile,
      .outer_bound = window.outer_bound,
      .inner_bound = window.inner_bound,
      .generation_stride = status.generation_stride,
      .declared_step_count = status.declared_step_count,
      .declared_step = source.failure.logical_step,
      .count_overflow_reason = source.failure.count_overflow_reason,
      .gather_reason = source.failure.invalid_index_reason,
      .reduce_reason = source.failure.reduce_overflow_reason,
      .profile_steps = profile_steps ? 1u : 0u,
      .profile_count = profile_steps ? status.declared_step_count : 0u,
      .profile_seed_first =
          profile_steps ? status.declared_steps[source.seed.first] : 0u,
      .action = action,
      .fold = fold,
  };
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
