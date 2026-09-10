#include "../../state/assembly.hpp"
#include "../arena.hpp"

#include "../../../memory/arena.hpp"
#include "../../state.hpp"
#include "../space.hpp"

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

} // namespace

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
    std::size_t offset_count = plan.offsets.size();
    std::size_t owner_count = plan.owners.size();
    std::vector<Step> order;
    order.reserve(steps.size());
    for (std::size_t index = 0u; index < steps.size(); ++index) {
      const PipelineBuildStep &step = steps[index];
      if (step.program == nullptr ||
          step.program->chunk_order.size() != step.program->chunks.size()) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const std::size_t chunks = step.program->chunk_order.size();
      if (chunks > plan.offsets.max_size() - offset_count ||
          chunks > plan.owners.max_size() - owner_count) {
        return Status::fail(Reason::PipelineCapacity);
      }
      plan.steps[index] = offset_count;
      offset_count += chunks;
      owner_count += chunks;
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
          plan.summary.largest_outer_window =
              step.route == PipelineRoute::NestedSeed
                  ? static_cast<std::size_t>(step.iteration)
                  : std::numeric_limits<std::size_t>::max();
          plan.summary.largest_inner_iteration =
              step.route == PipelineRoute::NestedAction
                  ? static_cast<std::size_t>(step.iteration)
                  : std::numeric_limits<std::size_t>::max();
          plan.summary.largest_nested_phase = pipeline_nested_phase(step.route);
          plan.summary.largest_chunk = ordinal;
        }
      }
      order.push_back(Step{.words = words, .index = index});
    }
    plan.steps.back() = offset_count;
    // The validation pass has sealed every prefix. Materialize each parallel
    // table once instead of reallocating and copying it at each step.
    plan.offsets.resize(offset_count);
    plan.owners.resize(owner_count);

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
        plan.summary.peak_outer_window =
            step.route == PipelineRoute::NestedSeed
                ? static_cast<std::size_t>(step.iteration)
                : std::numeric_limits<std::size_t>::max();
        plan.summary.peak_inner_iteration =
            step.route == PipelineRoute::NestedAction
                ? static_cast<std::size_t>(step.iteration)
                : std::numeric_limits<std::size_t>::max();
        plan.summary.peak_nested_phase = pipeline_nested_phase(step.route);
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
