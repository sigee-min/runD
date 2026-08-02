#pragma once

#include "../plan.hpp"
#include "../plan/local.hpp"
#include "../prepared/template_registry.hpp"
#include "../reset/model.hpp"
#include "../step/map/stride.hpp"
#include "run.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail::backend_template_plan {

using StepStructurePlan = rund::AccelCheck (*)(
    const KernelExecutionStep &, const rund::kernel::ComputePlan &,
    const BoundStep *, const KernelViewLayout *, std::uint64_t,
    PreparedKernelRouteReservation &) noexcept;

struct BackendShape final {
  std::uint64_t storage_alignment{1u};
  std::uint64_t max_dispatch_groups{};
  std::uint64_t reset_dispatch_window{};
  std::uint64_t template_capacity{};
  std::uint64_t route_header_bytes{};
  std::uint64_t route_step_bytes{};
  std::uint64_t route_inline_step_capacity{};
  std::uint64_t template_header_bytes{};
  std::uint64_t template_step_bytes{};
  std::uint64_t template_step_capacity{
      std::numeric_limits<std::uint64_t>::max()};
  StepStructurePlan plan_step{};
};

[[nodiscard]] inline bool add(std::uint64_t &target,
                              const std::uint64_t value) noexcept {
  return rund::kernel::checked::add(target, value, target);
}

[[nodiscard]] inline bool product(const std::uint64_t left,
                                  const std::uint64_t right,
                                  std::uint64_t &out) noexcept {
  return rund::kernel::checked::mul(left, right, out);
}

// Public planning and runtime specialization share the emitter-derived owner
// in step/map/stride.hpp. Backend controlled/private wrappers extend this
// scalar with their own shared allocation-free authorities.
[[nodiscard]] inline bool
map_source_upper(const KernelExecutionStep &step,
                 const rund::kernel::ComputePlan &plan, std::uint64_t &retained,
                 std::uint64_t &transient) noexcept {
  // Map edits are bounded by kMaxComputeBindingCount and frozen in a stack
  // array. The only heap owner is the final retained source string.
  transient = 0u;
  return MapSpecializedSourceUpperBytes(step.artifact, plan, retained);
}

[[nodiscard]] inline bool
same_plan(const rund::kernel::ComputePlan &left,
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

[[nodiscard]] inline bool
same_windows(const DispatchWindowStorage &left,
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

[[nodiscard]] inline bool
same_layout(const KernelViewLayout *const left,
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

[[nodiscard]] inline bool
same_layout(const KernelScratchLayout *const left,
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

[[nodiscard]] inline bool
same_map_binding_identity(const PreparedKernelProgramBindingIdentity &left,
                          const PreparedKernelProgramBindingIdentity &right,
                          const std::uint64_t alignment) noexcept {
  return alignment != 0u && left.element_bytes == right.element_bytes &&
         left.stride_bytes == right.stride_bytes && left.count == right.count &&
         left.usage == right.usage &&
         left.offset_bytes % alignment == right.offset_bytes % alignment;
}

[[nodiscard]] inline bool
same_program_map_specialization(const KernelExecution &execution,
                                const PreparedKernelProgramRoute &left,
                                const PreparedKernelProgramRoute &right,
                                const std::uint64_t alignment) noexcept {
  if (alignment == 0u ||
      left.program_bindings.size() != execution.graph_roles.size() ||
      right.program_bindings.size() != execution.graph_roles.size()) {
    return false;
  }
  for (const KernelExecutionStep &step : execution.steps) {
    if (step.kind() != rund::kernel::NodeKind::Map) {
      continue;
    }
    const auto &accesses = step.artifact.metadata.binding_accesses;
    if (!step.graph_binding_indices_ok ||
        accesses.size() > step.graph_binding_indices.size()) {
      return false;
    }
    for (std::size_t local = 0u; local < accesses.size(); ++local) {
      const std::uint64_t binding = step.graph_binding_indices[local];
      if (binding >= left.program_bindings.size() ||
          !same_map_binding_identity(left.program_bindings[binding],
                                     right.program_bindings[binding],
                                     alignment)) {
        return false;
      }
    }
  }
  return true;
}

struct MapSpecializationFingerprint final {
  std::uint64_t hi{0x72756e442e6d6170ull};
  std::uint64_t lo{0x2e7370656369616cull};
  bool ok{};
};

inline void mix_map_specialization(std::uint64_t &hash,
                                   const std::uint64_t value) noexcept {
  hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
}

inline void mix_map_specialization(
    MapSpecializationFingerprint &fingerprint,
    const PreparedKernelProgramBindingIdentity &identity) noexcept {
  mix_map_specialization(fingerprint.hi, identity.offset_bytes);
  mix_map_specialization(fingerprint.lo, identity.element_bytes);
  mix_map_specialization(fingerprint.hi, identity.stride_bytes);
  mix_map_specialization(fingerprint.lo, identity.count);
  mix_map_specialization(fingerprint.hi, identity.usage);
}

[[nodiscard]] inline MapSpecializationFingerprint
program_map_specialization_fingerprint(
    const KernelExecution &execution,
    const PreparedKernelProgramRoute &route) noexcept {
  MapSpecializationFingerprint fingerprint{};
  if (route.program_bindings.size() != execution.graph_roles.size()) {
    return fingerprint;
  }
  for (std::size_t step_index = 0u; step_index < execution.steps.size();
       ++step_index) {
    const KernelExecutionStep &step = execution.steps[step_index];
    if (step.kind() != rund::kernel::NodeKind::Map) {
      continue;
    }
    const auto &metadata = step.artifact.metadata;
    const auto &accesses = metadata.binding_accesses;
    if (!step.graph_binding_indices_ok ||
        accesses.size() > step.graph_binding_indices.size()) {
      return fingerprint;
    }
    mix_map_specialization(fingerprint.hi, step_index);
    mix_map_specialization(fingerprint.lo, metadata.read_count);
    mix_map_specialization(fingerprint.hi, metadata.write_count);
    for (const rund::kernel::ComputeBindingAccess selected :
         {rund::kernel::ComputeBindingAccess::Read,
          rund::kernel::ComputeBindingAccess::Write}) {
      for (std::size_t local = 0u; local < accesses.size(); ++local) {
        if (accesses[local] != selected) {
          continue;
        }
        const std::uint64_t binding = step.graph_binding_indices[local];
        if (binding >= route.program_bindings.size()) {
          return fingerprint;
        }
        mix_map_specialization(fingerprint, route.program_bindings[binding]);
      }
    }
  }
  fingerprint.ok = true;
  return fingerprint;
}

[[nodiscard]] inline MapSpecializationFingerprint
runtime_map_specialization_fingerprint(const BackendRun &run) noexcept {
  MapSpecializationFingerprint fingerprint{};
  if (run.steps == nullptr || run.step_count == 0u) {
    return fingerprint;
  }
  for (std::size_t step_index = 0u; step_index < run.step_count; ++step_index) {
    const BoundStep &step = run.steps[step_index];
    if (step.step == nullptr || step.planned == nullptr) {
      return fingerprint;
    }
    if (step.step->kind() != rund::kernel::NodeKind::Map) {
      continue;
    }
    const rund::kernel::BindingSet bindings = MapBindingFor(step);
    const auto &metadata = step.step->artifact.metadata;
    if (!bindings.ok || bindings.resident_inputs.count != metadata.read_count ||
        bindings.resident_outputs.count != metadata.write_count) {
      return fingerprint;
    }
    mix_map_specialization(fingerprint.hi, step_index);
    mix_map_specialization(fingerprint.lo, metadata.read_count);
    mix_map_specialization(fingerprint.hi, metadata.write_count);
    for (const rund::kernel::ResidentBindingRange range :
         {bindings.resident_inputs, bindings.resident_outputs}) {
      for (std::uint64_t index = 0u; index < range.count; ++index) {
        const rund::kernel::ResidentBufferRef *const ref = range.ref(index);
        if (ref == nullptr) {
          return fingerprint;
        }
        mix_map_specialization(fingerprint,
                               PreparedKernelProgramBindingIdentity{
                                   .offset_bytes = ref->offset_bytes,
                                   .element_bytes = ref->element_bytes,
                                   .stride_bytes = ref->stride_bytes,
                                   .count = ref->count,
                                   .usage = ref->usage,
                               });
      }
    }
  }
  fingerprint.ok = true;
  return fingerprint;
}

// Public planning and private materialization must partition immutable
// templates identically. A Kernel token owner is used only as a
// collision-safe equality witness; fingerprints remain entirely semantic.
// All route fields that private same_template() can observe before native
// compilation are compared here. Per-occurrence recurrence fields are route
// state and deliberately do not split an immutable Program template.
[[nodiscard]] inline bool
same_program_template(const KernelExecution &execution,
                      const PreparedKernelProgramRoute &left,
                      const PreparedKernelProgramRoute &right,
                      const std::uint64_t storage_alignment) noexcept {
  const rund::AccelKernel *const a = left.kernel;
  const rund::AccelKernel *const b = right.kernel;
  return storage_alignment != 0u && a != nullptr && b != nullptr &&
         a->owner != nullptr && a->owner == b->owner &&
         a->kernel_id == b->kernel_id && a->context_id == b->context_id &&
         a->graph_id_hi == b->graph_id_hi && a->graph_id_lo == b->graph_id_lo &&
         a->node_count == b->node_count && a->api == b->api &&
         a->scalar == b->scalar && a->domain == b->domain &&
         left.tile_count == right.tile_count &&
         same_layout(left.views, right.views) &&
         same_layout(left.scratch, right.scratch) &&
         same_program_map_specialization(execution, left, right,
                                         storage_alignment);
}

[[nodiscard]] inline bool
same_ref_layout(const rund::kernel::ResidentBindingRange &left,
                const rund::kernel::ResidentBindingRange &right,
                const std::uint64_t alignment) noexcept {
  if (alignment == 0u || left.count != right.count) {
    return false;
  }
  for (std::uint64_t index = 0u; index < left.count; ++index) {
    const rund::kernel::ResidentBufferRef *const a = left.ref(index);
    const rund::kernel::ResidentBufferRef *const b = right.ref(index);
    if (a == nullptr || b == nullptr || a->element_bytes != b->element_bytes ||
        a->stride_bytes != b->stride_bytes || a->count != b->count ||
        a->usage != b->usage ||
        a->offset_bytes % alignment != b->offset_bytes % alignment) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool
same_map_layout(const BoundStep &left, const BoundStep &right,
                const std::uint64_t alignment) noexcept {
  const rund::kernel::BindingSet a = MapBindingFor(left);
  const rund::kernel::BindingSet b = MapBindingFor(right);
  return a.ok && b.ok && a.input_buffer_count == b.input_buffer_count &&
         a.output_buffer_count == b.output_buffer_count &&
         same_ref_layout(a.resident_inputs, b.resident_inputs, alignment) &&
         same_ref_layout(a.resident_outputs, b.resident_outputs, alignment);
}

[[nodiscard]] inline bool
same_template(const BackendRun &left, const BackendRun &right,
              const std::uint64_t alignment) noexcept {
  if (left.pick == nullptr || right.pick == nullptr ||
      left.pick != right.pick || left.steps == nullptr ||
      right.steps == nullptr || left.step_count == 0u ||
      left.step_count != right.step_count ||
      !same_layout(left.views, right.views) ||
      !same_layout(left.scratch, right.scratch)) {
    return false;
  }
  for (std::size_t index = 0u; index < left.step_count; ++index) {
    const BoundStep &a = left.steps[index];
    const BoundStep &b = right.steps[index];
    if (a.step == nullptr || b.step == nullptr || a.step != b.step ||
        a.planned == nullptr || b.planned == nullptr ||
        a.planned->domain != b.planned->domain ||
        a.planned->artifact != b.planned->artifact ||
        !same_plan(a.planned->plan, b.planned->plan) ||
        !same_windows(a.planned->windows, b.planned->windows) ||
        a.control.active() != b.control.active()) {
      return false;
    }
    if (a.step->kind() == rund::kernel::NodeKind::Map &&
        !same_map_layout(a, b, alignment)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline std::uint64_t
primitive_pass_count(const KernelExecutionStep &step) noexcept {
  switch (step.kind()) {
  case rund::kernel::NodeKind::Map:
    return 1u;
  case rund::kernel::NodeKind::Scan:
    return step.operation.get<operation::Scan>().plan.pass_count;
  case rund::kernel::NodeKind::SegmentedScan:
    return step.operation.get<operation::SegmentedScan>().plan.pass_count;
  case rund::kernel::NodeKind::SegmentedReduce:
    return step.operation.get<operation::SegmentedReduce>().plan.pass_count;
  case rund::kernel::NodeKind::Sort:
    return step.operation.get<operation::Sort>().plan.radix_pass_count;
  case rund::kernel::NodeKind::Compact:
    return step.operation.get<operation::Compact>().plan.pass_count;
  case rund::kernel::NodeKind::Gather:
    return step.operation.get<operation::Gather>().plan.pass_count;
  case rund::kernel::NodeKind::Histogram:
    return step.operation.get<operation::Histogram>().plan.pass_count;
  case rund::kernel::NodeKind::Partition:
    return step.operation.get<operation::Partition>().plan.pass_count;
  case rund::kernel::NodeKind::Reduce:
    return step.operation.get<operation::Reduce>().plan.pass_count;
  case rund::kernel::NodeKind::Scatter:
    return step.operation.get<operation::Scatter>().plan.pass_count;
  case rund::kernel::NodeKind::ScatterReduce:
    return step.operation.get<operation::ScatterReduce>().plan.pass_count;
  case rund::kernel::NodeKind::Stencil:
    return step.operation.get<operation::Stencil>().plan.pass_count;
  case rund::kernel::NodeKind::Transform:
    return step.operation.get<operation::Transform>().plan.pass_count;
  case rund::kernel::NodeKind::Matrix:
    return step.operation.get<operation::Matrix>().plan.pass_count;
  case rund::kernel::NodeKind::Factor:
    return step.operation.get<operation::Factor>().plan.pass_count;
  case rund::kernel::NodeKind::Solve:
    return step.operation.get<operation::Solve>().plan.pass_count;
  case rund::kernel::NodeKind::Spectrum:
    return step.operation.get<operation::Spectrum>().plan.pass_count;
  }
  return 0u;
}

[[nodiscard]] inline rund::AccelCheck
plan(const BackendRun &run, const BackendShape shape,
     PreparedKernelRouteReservation &reservation) noexcept {
  reservation = {};
  if (run.pick == nullptr || run.steps == nullptr || run.step_count == 0u ||
      shape.storage_alignment == 0u || shape.max_dispatch_groups == 0u ||
      shape.reset_dispatch_window == 0u || shape.template_capacity == 0u ||
      shape.route_header_bytes == 0u || shape.route_step_bytes == 0u ||
      shape.route_inline_step_capacity == 0u ||
      shape.template_header_bytes == 0u || shape.template_step_bytes == 0u ||
      shape.template_step_capacity == 0u || shape.plan_step == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  std::uint64_t route_steps = 0u;
  std::uint64_t template_steps = 0u;
  if ((run.step_count > shape.route_inline_step_capacity &&
       !product(run.step_count, shape.route_step_bytes, route_steps)) ||
      !product(run.step_count, shape.template_step_bytes, template_steps)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  reservation.route_host_bytes = shape.route_header_bytes;
  reservation.template_host_bytes = shape.template_header_bytes;
  if (!add(reservation.route_host_bytes, route_steps) ||
      !add(reservation.template_host_bytes, template_steps)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  reservation.route_step_count = run.step_count;
  reservation.template_step_count = run.step_count;
  reservation.template_step_capacity = shape.template_step_capacity;
  reservation.template_capacity = shape.template_capacity;
  reservation.dispatch_count = run.final_dispatch_count;
  reservation.route_native_allocation_count = 1u;
  reservation.template_native_allocation_count = 1u;
  if (reservation.dispatch_count == 0u) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }

  if (run.resets != nullptr) {
    for (const BoundReset &reset : *run.resets) {
      if (!add(reservation.reset_dispatch_count,
               reset::Commands(reset.ref.count, shape.reset_dispatch_window))) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
    }
  }

  for (std::size_t index = 0u; index < run.step_count; ++index) {
    const BoundStep &bound = run.steps[index];
    if (bound.step == nullptr || bound.planned == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const std::uint64_t passes = primitive_pass_count(*bound.step);
    const rund::AccelCheck structured =
        shape.plan_step(*bound.step, bound.planned->plan, &bound, run.views,
                        shape.max_dispatch_groups, reservation);
    if (passes == 0u || !structured.ok) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    if (bound.step->kind() != rund::kernel::NodeKind::Map) {
      if (!add(reservation.route_native_bytes,
               bound.planned->plan.staging_bytes)) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      continue;
    }

    const rund::kernel::ComputePlan &map = bound.planned->plan;
    const std::uint64_t windows = bound.planned->windows.size();
    const rund::kernel::BindingSet bindings = MapBindingFor(bound);
    if (!map.ok || windows == 0u || !bindings.ok ||
        map.input_buffer_count != bindings.resident_inputs.count ||
        map.output_buffer_count != bindings.resident_outputs.count) {
      return rund::AccelCheck{false, "compute_plan_invalid"};
    }
    std::uint64_t window_bytes = 0u;
    if (!product(windows, sizeof(rund::kernel::ComputeDispatchWindow),
                 window_bytes) ||
        !add(reservation.route_host_bytes, window_bytes) ||
        !add(reservation.route_native_bytes, map.param_bytes) ||
        !add(reservation.route_native_allocation_count, 1u)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    const bool controlled = bound.control.active() ||
                            !bound.step->artifact.metadata.read_routes.empty();
    if (controlled) {
      std::uint64_t indirect_bytes = 0u;
      if (!product(windows, 4u * sizeof(std::uint32_t), indirect_bytes) ||
          !add(reservation.route_native_bytes, indirect_bytes) ||
          !add(reservation.route_native_bytes, 2u * sizeof(std::uint32_t)) ||
          !add(reservation.route_native_allocation_count, 2u)) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
    }
  }
  return rund::AccelCheck{true, "ok"};
}

[[nodiscard]] inline rund::AccelCheck
plan_program(const KernelExecution &execution,
             const PreparedKernelProgramRoute &route, const BackendShape shape,
             PreparedKernelRouteReservation &reservation) noexcept {
  reservation = {};
  if (!execution.admission.check.ok || execution.steps.empty() ||
      route.kernel == nullptr || route.tile_count == 0u ||
      route.route_copies == 0u || shape.storage_alignment == 0u ||
      shape.max_dispatch_groups == 0u || shape.reset_dispatch_window == 0u ||
      shape.template_capacity == 0u || shape.route_header_bytes == 0u ||
      shape.route_step_bytes == 0u || shape.route_inline_step_capacity == 0u ||
      shape.template_header_bytes == 0u || shape.template_step_bytes == 0u ||
      shape.template_step_capacity == 0u || shape.plan_step == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  std::uint64_t route_steps = 0u;
  std::uint64_t template_steps = 0u;
  std::uint64_t view_bytes = 0u;
  std::uint64_t scratch_bytes = 0u;
  std::uint64_t reset_bytes = 0u;
  if ((execution.steps.size() > shape.route_inline_step_capacity &&
       !product(execution.steps.size(), shape.route_step_bytes, route_steps)) ||
      !product(execution.steps.size(), shape.template_step_bytes,
               template_steps) ||
      !product(route.views == nullptr ? 0u : route.views->size(),
               sizeof(KernelViewSlot), view_bytes) ||
      !product(route.scratch == nullptr ? 0u : route.scratch->size(),
               sizeof(KernelScratchPage), scratch_bytes) ||
      !product(execution.resets.size(), sizeof(BoundReset), reset_bytes)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  reservation.route_host_bytes = shape.route_header_bytes;
  reservation.template_host_bytes = shape.template_header_bytes;
  if (!add(reservation.route_host_bytes, route_steps) ||
      !add(reservation.route_host_bytes, view_bytes) ||
      !add(reservation.route_host_bytes, scratch_bytes) ||
      !add(reservation.route_host_bytes, reset_bytes) ||
      !add(reservation.template_host_bytes, template_steps)) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  reservation.route_step_count = execution.steps.size();
  reservation.template_step_count = execution.steps.size();
  reservation.template_step_capacity = shape.template_step_capacity;
  reservation.template_capacity = shape.template_capacity;
  reservation.route_native_allocation_count = 1u;
  reservation.template_native_allocation_count = 1u;

  for (const ResetPlan &reset : execution.resets) {
    if (reset.binding >= execution.graph_shapes.size() ||
        !add(reservation.reset_dispatch_count,
             reset::Commands(execution.graph_shapes[reset.binding].count,
                             shape.reset_dispatch_window))) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }

  const rund::AccelRun run{.tile_count = route.tile_count};
  for (std::size_t index = 0u; index < execution.steps.size(); ++index) {
    const KernelExecutionStep &step = execution.steps[index];
    // Map planning specializes the run tile/domain metadata directly. Every
    // primitive already owns its canonical operation plan; routing it through
    // Map's ExecutionMetadata authority zeroes the primitive op hash and can
    // reject an otherwise admitted Program during public preflight.
    const rund::kernel::ComputePlan step_plan =
        step.kind() == rund::kernel::NodeKind::Map
            ? PlanStep(execution, step, run, index)
            : BuildPlannedStep(execution, step, run, index).plan;
    if (!step_plan.ok || step_plan.dispatch_count == 0u ||
        !add(reservation.dispatch_count, step_plan.dispatch_count)) {
      return rund::AccelCheck{false, step_plan.reason == nullptr
                                         ? "compute_pipeline_capacity"
                                         : step_plan.reason};
    }
    const std::uint64_t passes = primitive_pass_count(step);
    const rund::AccelCheck structured =
        shape.plan_step(step, step_plan, nullptr, route.views,
                        shape.max_dispatch_groups, reservation);
    if (passes == 0u || !structured.ok) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    if (step.kind() != rund::kernel::NodeKind::Map) {
      if (!add(reservation.route_native_bytes, step_plan.staging_bytes)) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      continue;
    }

    const rund::kernel::ComputePlan map = step_plan;
    if (!map.ok || map.dispatch_count == 0u ||
        map.dispatch_count > std::numeric_limits<std::uint32_t>::max()) {
      return rund::AccelCheck{
          false, map.reason == nullptr ? "compute_plan_invalid" : map.reason};
    }
    std::uint64_t window_bytes = 0u;
    if (!product(map.dispatch_count,
                 sizeof(rund::kernel::ComputeDispatchWindow), window_bytes) ||
        !add(reservation.route_host_bytes, window_bytes) ||
        !add(reservation.route_native_bytes, map.param_bytes) ||
        !add(reservation.route_native_allocation_count, 1u)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }

    const bool controlled = step.control.has_count() ||
                            step.control.has_predicate() ||
                            !step.artifact.metadata.read_routes.empty();
    if (controlled) {
      std::uint64_t indirect_bytes = 0u;
      if (!product(map.dispatch_count, 4u * sizeof(std::uint32_t),
                   indirect_bytes) ||
          !add(reservation.route_native_bytes, indirect_bytes) ||
          !add(reservation.route_native_bytes, 2u * sizeof(std::uint32_t)) ||
          !add(reservation.route_native_allocation_count, 2u)) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
    }
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace rund::node::accel::detail::backend_template_plan
