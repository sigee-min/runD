#include "bindings.hpp"

#include "../../context/shared.hpp"
#include "../reset/overlap.hpp"
#include "../reset/proof.hpp"
#include "shape.hpp"

#include <accel/check.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/context/value.hpp>
#include <accel/kernel/run/binding.hpp>
#include <kernel/core/checked.hpp>

#include <new>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace rund::node::accel::detail {

RunBindBuild BuildRunBinds(const KernelExecution &execution,
                           const rund::AccelRun &run) {
  RunBindBuild result{};
  result.binds.reserve(run.binding_count);
  if (!result.binds.ok) {
    return result;
  }
  for (std::uint64_t index = 0u; index < run.binding_count; ++index) {
    const rund::AccelRunBinding &binding = run.bindings[index];
    if (!RoleMatches(execution.graph_roles[static_cast<std::size_t>(index)],
                     binding.role) ||
        binding.buffer == nullptr) {
      return result;
    }
    std::shared_ptr<void> backend_handle{};
    const rund::AccelCheck check = ValidateAccelBufferForSupport(
        execution.context_admission, *binding.buffer, backend_handle);
    if (!check.ok) {
      result.reason = "accel_kernel_buffer_owner_mismatch";
      return result;
    }
    const rund::AccelBufferDesc &expected =
        execution.graph_shapes[static_cast<std::size_t>(index)];
    const std::uint64_t element_bytes = binding.element_bytes == 0u
                                            ? expected.scalar_width_bytes
                                            : binding.element_bytes;
    const std::uint64_t element_count =
        binding.element_count == 0u ? expected.count : binding.element_count;
    const std::uint64_t stride_bytes =
        binding.stride_bytes == 0u ? element_bytes : binding.stride_bytes;
    const std::uint64_t alignment =
        binding.alignment == 0u ? element_bytes : binding.alignment;
    const std::uint64_t tail_count =
        element_count == 0u ? 0u : element_count - 1u;
    std::uint64_t tail = 0u;
    std::uint64_t envelope = 0u;
    const bool envelope_ok =
        rund::kernel::checked::mul(tail_count, stride_bytes, tail) &&
        rund::kernel::checked::add(binding.offset_bytes, tail, envelope) &&
        rund::kernel::checked::add(envelope, element_bytes, envelope);
    const bool internal =
        execution.graph_visibilities[static_cast<std::size_t>(index)] ==
        rund::GraphBufferVisibility::Internal;
    if ((!internal &&
         binding.buffer->scalar_width_bytes != expected.scalar_width_bytes) ||
        element_bytes != expected.scalar_width_bytes ||
        element_count != expected.count || element_count == 0u ||
        stride_bytes < element_bytes || alignment == 0u ||
        binding.offset_bytes % alignment != 0u ||
        stride_bytes % alignment != 0u || !envelope_ok ||
        envelope > binding.buffer->byte_extent ||
        !UsageCompatible(expected.usage, binding.buffer->usage)) {
      result.reason = "accel_kernel_buffer_shape_mismatch";
      return result;
    }
    const std::uint64_t representative =
        execution.graph_alias_representatives[static_cast<std::size_t>(index)];
    if (representative > index) {
      return result;
    }
    if (representative != index) {
      const rund::AccelRunBinding &first = run.bindings[representative];
      if (first.buffer == nullptr ||
          first.buffer->resident.id != binding.buffer->resident.id ||
          first.offset_bytes != binding.offset_bytes ||
          (first.element_count == 0u ? expected.count : first.element_count) !=
              element_count ||
          (first.stride_bytes == 0u ? element_bytes : first.stride_bytes) !=
              stride_bytes) {
        result.reason = "accel_kernel_buffer_alias_mismatch";
        return result;
      }
    }
    const rund::kernel::ResidentBufferRef ref{
        .id = binding.buffer->resident.id,
        .bytes = binding.buffer->byte_extent,
        .offset_bytes = binding.offset_bytes,
        .element_bytes = element_bytes,
        .stride_bytes = stride_bytes,
        .count = element_count,
        .usage = binding.role == rund::kernel::BufferRole::Read
                     ? rund::kernel::kResidentUsageRead
                     : rund::kernel::kResidentUsageWrite,
    };
    if (!result.binds.push(ref, std::move(backend_handle))) {
      return result;
    }
  }
  result.ok = result.binds.valid() && result.binds.size() == run.binding_count;
  return result;
}

ResetBindBuild BuildResetBinds(const KernelExecution &execution,
                               const rund::AccelRun &run,
                               const RunBinds &bindings) try {
  ResetBindBuild result{};
  if (execution.resets.empty()) {
    result.ok = true;
    result.reason = "ok";
    return result;
  }
  result.binds.reserve(execution.resets.size());
  std::tuple<std::uint32_t, std::uint64_t> prior{};
  for (std::size_t index = 0u; index < execution.resets.size(); ++index) {
    const ResetPlan reset = execution.resets[index];
    if (reset.binding >= run.binding_count ||
        reset.binding >= execution.graph_roles.size() || !reset.step.valid() ||
        !reset.last.valid() || reset.step.index >= execution.steps.size() ||
        reset.last.index >= execution.steps.size() ||
        reset.last.index < reset.step.index) {
      return result;
    }
    const auto order = std::tie(reset.step.index, reset.binding);
    if (index != 0u && order <= prior) {
      return result;
    }
    const rund::AccelRunBinding &source = run.bindings[reset.binding];
    if (source.buffer == nullptr ||
        source.role != rund::kernel::BufferRole::Write ||
        execution.graph_roles[reset.binding] !=
            rund::kernel::BufferRole::Write ||
        source.buffer->usage == rund::BufferUsage::ReadOnly ||
        !bindings.valid() || reset.binding >= bindings.size()) {
      return result;
    }
    const rund::kernel::ResidentBufferRef source_ref =
        bindings.refs()[reset.binding];
    std::shared_ptr<void> handle = bindings.handles()[reset.binding];
    if (source_ref.bytes == 0u || handle == nullptr) {
      return result;
    }
    const reset::Result proved =
        reset::Prove(reset::Project(source_ref, nullptr), source_ref.bytes);
    if (!proved.check.ok) {
      result.reason = proved.check.reason;
      return result;
    }
    std::optional<BoundReset> sealed = BoundReset::Seal(
        source_ref, std::move(handle), proved.range, reset.binding, reset.step,
        reset.last, execution.graph_visibilities[reset.binding] ==
                        rund::GraphBufferVisibility::External);
    if (!sealed.has_value()) {
      return result;
    }
    result.binds.push_back(std::move(*sealed));
    prior = order;
  }
  if (!reset::Compatible(result.binds)) {
    return result;
  }
  result.ok = result.binds.size() == execution.resets.size();
  if (result.ok) {
    result.reason = "ok";
  }
  return result;
} catch (const std::bad_alloc &) {
  return {};
} catch (const std::length_error &) {
  return {};
}

} // namespace rund::node::accel::detail
