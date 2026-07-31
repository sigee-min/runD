#include "arena.hpp"

#include "../../memory/arena.hpp"
#include "../../type.hpp"
#include "../state.hpp"
#include "compare.hpp"
#include "output.hpp"
#include "space.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <tuple>
#include <utility>
#include <vector>

namespace rund::compute::detail {
namespace {

struct Step final {
  std::uint64_t words{};
  std::size_t index{};
};

struct View final {
  std::uint64_t binding{};
  std::uint64_t bytes{};
  std::uint64_t span_bytes{};
  std::uint64_t backing_bytes{};
  std::uint64_t offset_bytes{};
  std::uint64_t stride_bytes{};
  std::uint64_t element_bytes{};
  std::uint64_t count{};
  std::uint64_t alignment{};
  std::size_t words{};
  std::size_t alignment_words{1u};
};

} // namespace

Status plan_pipeline_views(const DeviceState &device,
                           const std::span<const PipelineBuildStep> steps,
                           PipelineMemoryPlan &plan) noexcept {
  plan.views.clear();
  plan.view_slots.clear();
  plan.view_chunks.clear();
  plan.views.resize(steps.size());
  const Backend backend = device.backend;
  if (backend == Backend::Cpu) {
    return Status::success();
  }
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  const std::uint64_t alignment = accel->pick.caps.storage_alignment;
  const space::Bounds bounds = space::bounds(device);
  if (!kernel::ComputeStorageAlignmentValid(alignment) ||
      alignment > std::numeric_limits<std::size_t>::max() ||
      bounds.ordinary == 0u) {
    return Status::fail(Reason::PipelineInvalid);
  }
  try {
    std::uint64_t logical = 0u;
    std::vector<View> active;
    for (std::size_t pipeline_step = 0u; pipeline_step < steps.size();
         ++pipeline_step) {
      const PipelineBuildStep &declared = steps[pipeline_step];
      if (declared.program == nullptr) {
        return Status::fail(Reason::PipelineInvalid);
      }
      if (declared.iteration_bound > 1u && declared.iteration >= 3u) {
        if (pipeline_step < declared.iteration) {
          return Status::fail(Reason::PipelineInvalid);
        }
        const std::size_t first = pipeline_step - declared.iteration;
        const std::size_t phase =
            first + ((declared.iteration & 1u) != 0u ? 1u : 2u);
        if (phase >= pipeline_step) {
          return Status::fail(Reason::PipelineInvalid);
        }
        // Only exact two-bank phases share the prepared Job that consumes this
        // layout. Resident window controls carry an iteration-specific view;
        // those occurrences remain valid but must be planned below instead of
        // being rejected as a malformed recurrence.
        if (same_recurrence_phase(declared, steps[phase])) {
          if (!kernel::checked::add(
                  logical, static_cast<std::uint64_t>(plan.views[phase].size()),
                  logical)) {
            return Status::fail(Reason::PipelineCapacity);
          }
          continue;
        }
      }
      const ProgramState &program = *declared.program;
      if (program.graph_info.nodes.empty()) {
        if (!program.graph_bindings.empty()) {
          return Status::fail(Reason::GraphBindingInvalid);
        }
        continue;
      }
      auto projection = project_outputs(declared);
      if (!projection) {
        return Status::fail(projection.reason());
      }
      std::size_t binding_base = 0u;
      for (const graph::Node &node : program.graph_info.nodes) {
        if (binding_base > program.graph_bindings.size() ||
            node.accesses.size() >
                program.graph_bindings.size() - binding_base) {
          return Status::fail(Reason::GraphBindingInvalid);
        }
        active.clear();
        if (node.operation != graph::Operation::Map &&
            node.operation != graph::Operation::ScatterReduce) {
          active.reserve(node.accesses.size());
          for (std::size_t local = 0u; local < node.accesses.size(); ++local) {
            const std::size_t binding = binding_base + local;
            const GraphRunBinding route_binding =
                program.graph_bindings[binding];
            const graph::Access &access = node.accesses[local];
            if (access.resource == 0u ||
                route_binding.value_index != access.resource - 1u ||
                route_binding.value_index >=
                    program.graph_value_routes.size()) {
              return Status::fail(Reason::GraphBindingInvalid);
            }
            const GraphValueRoute route =
                program.graph_value_routes[route_binding.value_index];
            const PipelineBinding *view = nullptr;
            Type type{Type::U32};
            if (route.source == GraphBindSource::Input) {
              if (route.index >= declared.inputs.size() ||
                  route.index >= program.input_types.size()) {
                return Status::fail(Reason::GraphBindingInvalid);
              }
              view = &declared.inputs[route.index];
              type = program.input_types[route.index];
            } else if (route.source == GraphBindSource::Output) {
              if (route.index >= projection->physical_count ||
                  route.index >= program.output_types.size()) {
                return Status::fail(Reason::GraphBindingInvalid);
              }
              const std::uint32_t source =
                  projection->physical_sources[route.index];
              if (source == OutputProjection::unassigned ||
                  source >= declared.outputs.size()) {
                return Status::fail(Reason::GraphBindingInvalid);
              }
              view = &declared.outputs[source];
              type = program.output_types[route.index];
            }
            if (view == nullptr || view->count == 0u) {
              continue;
            }
            const bool strided = view->count > 1u && view->stride != 1u;
            std::uint64_t offset_bytes = 0u;
            if (!kernel::checked::mul(
                    static_cast<std::uint64_t>(view->offset),
                    static_cast<std::uint64_t>(view->element_bytes),
                    offset_bytes)) {
              return Status::fail(Reason::PipelineCapacity);
            }
            const bool vulkan_dense =
                backend == Backend::Vulkan && offset_bytes % alignment != 0u;
            if (!strided && !vulkan_dense) {
              continue;
            }
            std::uint64_t bytes = 0u;
            std::uint64_t stride_bytes = 0u;
            std::uint64_t span_bytes = 0u;
            if (view->element_bytes == 0u ||
                view->element_bytes != type_bytes(type) ||
                view->element_bytes % sizeof(std::uint32_t) != 0u ||
                !kernel::checked::mul(
                    static_cast<std::uint64_t>(view->count),
                    static_cast<std::uint64_t>(view->element_bytes), bytes) ||
                !kernel::checked::mul(
                    static_cast<std::uint64_t>(view->stride),
                    static_cast<std::uint64_t>(view->element_bytes),
                    stride_bytes) ||
                !kernel::checked::mul(
                    static_cast<std::uint64_t>(view->count - 1u), stride_bytes,
                    span_bytes) ||
                !kernel::checked::add(
                    span_bytes, static_cast<std::uint64_t>(view->element_bytes),
                    span_bytes)) {
              return Status::fail(Reason::GraphBindingInvalid);
            }
            const std::uint64_t words64 = bytes / sizeof(std::uint32_t);
            if (words64 == 0u ||
                words64 > std::numeric_limits<std::size_t>::max()) {
              return Status::fail(Reason::PipelineCapacity);
            }
            active.push_back(
                View{.binding = binding,
                     .bytes = bytes,
                     .span_bytes = span_bytes,
                     .backing_bytes = view->backing_bytes,
                     .offset_bytes = offset_bytes,
                     .stride_bytes = stride_bytes,
                     .element_bytes = view->element_bytes,
                     .count = view->count,
                     .alignment = view->alignment,
                     .words = static_cast<std::size_t>(words64),
                     .alignment_words = std::max<std::size_t>(
                         1u, view->element_bytes / sizeof(std::uint32_t))});
          }
        }
        binding_base += node.accesses.size();
        std::sort(active.begin(), active.end(),
                  [](const View &left, const View &right) {
                    if (left.bytes != right.bytes) {
                      return left.bytes > right.bytes;
                    }
                    return left.binding < right.binding;
                  });
        for (std::size_t rank = 0u; rank < active.size(); ++rank) {
          const View &view = active[rank];
          if (rank == plan.view_slots.size()) {
            plan.view_slots.push_back(PipelineMemoryPlan::ViewSlot{
                .words = view.words, .alignment_words = view.alignment_words});
          } else {
            plan.view_slots[rank].words =
                std::max(plan.view_slots[rank].words, view.words);
            plan.view_slots[rank].alignment_words = std::max(
                plan.view_slots[rank].alignment_words, view.alignment_words);
          }
          plan.views[pipeline_step].push_back(
              node::accel::detail::KernelViewSlot{
                  .binding = view.binding,
                  .slot = rank,
                  .bytes = view.bytes,
              });
          if (!kernel::checked::add(logical, 1u, logical)) {
            return Status::fail(Reason::PipelineCapacity);
          }
          const auto location =
              std::tuple{declared.logical_step, declared.iteration,
                         static_cast<std::size_t>(view.binding)};
          const auto largest =
              std::tuple{plan.summary.view_step, plan.summary.view_iteration,
                         plan.summary.view_binding};
          if (view.bytes > plan.summary.view_bytes ||
              (view.bytes == plan.summary.view_bytes && location < largest)) {
            plan.summary.view_bytes = view.bytes;
            plan.summary.view_span_bytes = view.span_bytes;
            plan.summary.view_backing_bytes = view.backing_bytes;
            plan.summary.view_offset_bytes = view.offset_bytes;
            plan.summary.view_stride_bytes = view.stride_bytes;
            plan.summary.view_element_bytes = view.element_bytes;
            plan.summary.view_count = view.count;
            plan.summary.view_alignment = view.alignment;
            plan.summary.view_step = declared.logical_step;
            plan.summary.view_iteration = declared.iteration;
            plan.summary.view_binding = view.binding;
          }
        }
      }
      if (binding_base != program.graph_bindings.size()) {
        return Status::fail(Reason::GraphBindingInvalid);
      }
    }
    const std::size_t byte_alignment =
        std::max(static_cast<std::size_t>(alignment), sizeof(std::uint32_t));
    if (byte_alignment % sizeof(std::uint32_t) != 0u) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t word_alignment = byte_alignment / sizeof(std::uint32_t);
    std::vector<std::size_t> placement;
    placement.reserve(plan.view_slots.size());
    for (std::size_t slot = 0u; slot < plan.view_slots.size(); ++slot) {
      const PipelineMemoryPlan::ViewSlot &view = plan.view_slots[slot];
      if (view.words == 0u || view.alignment_words == 0u ||
          (view.alignment_words & (view.alignment_words - 1u)) != 0u) {
        return Status::fail(Reason::PipelineInvalid);
      }
      placement.push_back(slot);
    }
    std::sort(placement.begin(), placement.end(),
              [&](const std::size_t left, const std::size_t right) {
                const PipelineMemoryPlan::ViewSlot &left_slot =
                    plan.view_slots[left];
                const PipelineMemoryPlan::ViewSlot &right_slot =
                    plan.view_slots[right];
                if (left_slot.words != right_slot.words) {
                  return left_slot.words > right_slot.words;
                }
                if (left_slot.alignment_words != right_slot.alignment_words) {
                  return left_slot.alignment_words > right_slot.alignment_words;
                }
                return left < right;
              });
    for (const std::size_t slot : placement) {
      const PipelineMemoryPlan::ViewSlot &view = plan.view_slots[slot];
      const std::size_t words = view.words;
      const std::size_t slot_alignment =
          std::max(word_alignment, view.alignment_words);
      if (words > bounds.storage) {
        return Status::fail(Reason::PipelineCapacity);
      }
      std::size_t selected = std::numeric_limits<std::size_t>::max();
      std::size_t selected_offset = 0u;
      std::size_t best_slack = std::numeric_limits<std::size_t>::max();
      if (words <= bounds.ordinary) {
        for (std::size_t owner = 0u; owner < plan.view_chunks.size(); ++owner) {
          if (plan.view_chunks[owner] > bounds.ordinary) {
            continue;
          }
          std::size_t offset = 0u;
          if (!space::align(plan.view_chunks[owner], slot_alignment, offset) ||
              offset > bounds.ordinary || words > bounds.ordinary - offset) {
            continue;
          }
          const std::size_t slack = bounds.ordinary - offset - words;
          if (std::tie(slack, owner) < std::tie(best_slack, selected)) {
            selected = owner;
            selected_offset = offset;
            best_slack = slack;
          }
        }
      }
      if (selected == std::numeric_limits<std::size_t>::max()) {
        selected = plan.view_chunks.size();
        selected_offset = 0u;
        plan.view_chunks.push_back(words);
      } else {
        plan.view_chunks[selected] = selected_offset + words;
      }
      plan.view_slots[slot].owner = selected;
      plan.view_slots[slot].offset_words = selected_offset;
    }
    for (const std::size_t words : plan.view_chunks) {
      std::uint64_t bytes = 0u;
      if (!kernel::checked::mul(
              static_cast<std::uint64_t>(words),
              static_cast<std::uint64_t>(sizeof(std::uint32_t)), bytes) ||
          !kernel::checked::add(plan.summary.prepared_bytes, bytes,
                                plan.summary.prepared_bytes)) {
        return Status::fail(Reason::PipelineCapacity);
      }
    }
    const std::uint64_t physical =
        static_cast<std::uint64_t>(plan.view_slots.size());
    std::uint64_t reused = 0u;
    if (logical < physical ||
        !kernel::checked::add(plan.summary.reuse_count, logical - physical,
                              reused)) {
      return Status::fail(Reason::PipelineCapacity);
    }
    plan.summary.reuse_count = reused;
    return Status::success();
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::PipelineCapacity);
  }
}

Status plan_pipeline_arena(const DeviceState &device,
                           const std::span<const PipelineBuildStep> steps,
                           PipelineMemoryPlan &plan) noexcept {
  try {
    const space::Bounds bounds = space::bounds(device);
    if (bounds.ordinary == 0u) {
      return Status::fail(Reason::PipelineCapacity);
    }
    plan.steps.assign(steps.size() + 1u, 0u);
    std::uint64_t logical_chunks = 0u;
    std::vector<Step> order;
    order.reserve(steps.size());
    for (std::size_t index = 0u; index < steps.size(); ++index) {
      const PipelineBuildStep &step = steps[index];
      if (step.program == nullptr ||
          step.program->chunk_order.size() != step.program->chunks.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      plan.steps[index] = plan.offsets.size();
      plan.offsets.resize(plan.offsets.size() +
                          step.program->chunk_order.size());
      plan.owners.resize(plan.owners.size() + step.program->chunk_order.size());
      std::uint64_t words = 0u;
      for (const std::uint32_t ordinal : step.program->chunk_order) {
        if (ordinal >= step.program->chunks.size()) {
          return Status::fail(Reason::PipelineInvalid);
        }
        const std::size_t count = step.program->chunks[ordinal].count;
        std::uint64_t bytes = 0u;
        if (count == 0u || count > bounds.storage ||
            !kernel::checked::mul(static_cast<std::uint64_t>(count),
                                  memory::Word, bytes) ||
            !kernel::checked::add(words, static_cast<std::uint64_t>(count),
                                  words) ||
            !kernel::checked::add(logical_chunks, 1u, logical_chunks)) {
          return Status::fail(Reason::PipelineCapacity);
        }
        const auto location = std::tuple{step.logical_step, step.iteration,
                                         static_cast<std::size_t>(ordinal)};
        const auto largest = std::tuple{plan.summary.largest_step,
                                        plan.summary.largest_iteration,
                                        plan.summary.largest_chunk};
        if (bytes > plan.summary.largest_bytes ||
            (bytes == plan.summary.largest_bytes && location < largest)) {
          plan.summary.largest_bytes = bytes;
          plan.summary.largest_step = step.logical_step;
          plan.summary.largest_iteration = step.iteration;
          plan.summary.largest_chunk = ordinal;
        }
      }
      order.push_back(Step{.words = words, .index = index});
    }
    plan.steps.back() = plan.offsets.size();

    std::sort(order.begin(), order.end(),
              [&](const Step left, const Step right) {
                const PipelineBuildStep &left_step = steps[left.index];
                const PipelineBuildStep &right_step = steps[right.index];
                if (left.words != right.words) {
                  return left.words > right.words;
                }
                return std::tie(left_step.logical_step, left_step.iteration,
                                left.index) < std::tie(right_step.logical_step,
                                                       right_step.iteration,
                                                       right.index);
              });

    std::vector<std::size_t> used;
    for (const Step planned : order) {
      const PipelineBuildStep &step = steps[planned.index];
      used.assign(plan.chunks.size(), 0u);
      const std::size_t base = plan.steps[planned.index];
      for (std::size_t rank = 0u; rank < step.program->chunk_order.size();
           ++rank) {
        const std::size_t chunk = step.program->chunk_order[rank];
        const std::size_t count = step.program->chunks[chunk].count;
        std::size_t selected = std::numeric_limits<std::size_t>::max();
        std::size_t selected_offset = 0u;
        std::size_t best_growth = std::numeric_limits<std::size_t>::max();
        std::size_t best_slack = std::numeric_limits<std::size_t>::max();
        for (std::size_t owner = 0u; owner < plan.chunks.size(); ++owner) {
          const bool dedicated =
              count > bounds.ordinary || plan.chunks[owner] > bounds.ordinary;
          if (dedicated) {
            if (used[owner] != 0u) {
              continue;
            }
            const std::size_t growth =
                count > plan.chunks[owner] ? count - plan.chunks[owner] : 0u;
            const std::size_t slack =
                count < plan.chunks[owner] ? plan.chunks[owner] - count : 0u;
            if (std::tie(growth, slack, owner) <
                std::tie(best_growth, best_slack, selected)) {
              selected = owner;
              selected_offset = 0u;
              best_growth = growth;
              best_slack = slack;
            }
            continue;
          }
          std::size_t offset = 0u;
          if (!space::align(used[owner], offset) || offset > bounds.ordinary ||
              count > bounds.ordinary - offset) {
            continue;
          }
          const std::size_t end = offset + count;
          const std::size_t growth =
              end > plan.chunks[owner] ? end - plan.chunks[owner] : 0u;
          const std::size_t slack = end < plan.chunks[owner]
                                        ? plan.chunks[owner] - end
                                        : bounds.ordinary - end;
          if (std::tie(growth, slack, owner) <
              std::tie(best_growth, best_slack, selected)) {
            selected = owner;
            selected_offset = offset;
            best_growth = growth;
            best_slack = slack;
          }
        }
        if (selected == std::numeric_limits<std::size_t>::max()) {
          selected = plan.chunks.size();
          selected_offset = 0u;
          plan.chunks.push_back(count);
          used.push_back(count);
        } else {
          const std::size_t end = selected_offset + count;
          plan.chunks[selected] = std::max(plan.chunks[selected], end);
          used[selected] = end;
        }
        plan.owners[base + rank] = selected;
        plan.offsets[base + rank] = selected_offset;
      }
    }

    std::vector<std::size_t> touched = std::move(used);
    std::vector<std::size_t> extents(plan.chunks.size());
    std::uint64_t peak_words = 0u;
    bool peak_set = false;
    for (std::size_t index = 0u; index < steps.size(); ++index) {
      const PipelineBuildStep &step = steps[index];
      const std::size_t base = plan.steps[index];
      touched.clear();
      for (std::size_t rank = 0u; rank < step.program->chunk_order.size();
           ++rank) {
        const std::size_t chunk = step.program->chunk_order[rank];
        const std::size_t owner = plan.owners[base + rank];
        const std::size_t offset = plan.offsets[base + rank];
        const std::size_t count = step.program->chunks[chunk].count;
        if (owner >= extents.size() ||
            count > std::numeric_limits<std::size_t>::max() - offset) {
          return Status::fail(Reason::PipelineInvalid);
        }
        if (extents[owner] == 0u) {
          touched.push_back(owner);
        }
        extents[owner] = std::max(extents[owner], offset + count);
      }
      std::uint64_t words = 0u;
      for (const std::size_t owner : touched) {
        const std::size_t extent = extents[owner];
        extents[owner] = 0u;
        if (!kernel::checked::add(words, static_cast<std::uint64_t>(extent),
                                  words)) {
          return Status::fail(Reason::PipelineCapacity);
        }
      }
      const auto location = std::tie(step.logical_step, step.iteration);
      const auto peak =
          std::tie(plan.summary.peak_step, plan.summary.peak_iteration);
      if (!peak_set || words > peak_words ||
          (words == peak_words && location < peak)) {
        peak_set = true;
        peak_words = words;
        plan.summary.peak_step = step.logical_step;
        plan.summary.peak_iteration = step.iteration;
      }
    }

    for (const std::size_t count : plan.chunks) {
      std::uint64_t bytes = 0u;
      if (count == 0u ||
          !kernel::checked::mul(static_cast<std::uint64_t>(count), memory::Word,
                                bytes) ||
          !kernel::checked::add(plan.summary.transient_bytes, bytes,
                                plan.summary.transient_bytes)) {
        return Status::fail(Reason::PipelineCapacity);
      }
    }
    if (plan.chunks.size() > logical_chunks) {
      return Status::fail(Reason::PipelineInvalid);
    }
    plan.summary.reuse_count =
        logical_chunks - static_cast<std::uint64_t>(plan.chunks.size());
    return Status::success();
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::PipelineCapacity);
  }
}

} // namespace rund::compute::detail
