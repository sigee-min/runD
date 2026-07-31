#pragma once

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::node::accel::detail {

inline constexpr std::size_t BatchMapCapacity = 64u;
inline constexpr std::size_t BatchMapBindingCapacity = 16u;

struct BatchMapView final {
  std::uint64_t kernel = 0u;
  std::uint64_t tiles = 0u;
  std::array<std::uint64_t, BatchMapBindingCapacity> inputs{};
  std::array<std::uint64_t, BatchMapBindingCapacity> outputs{};
  std::array<std::uint64_t, BatchMapBindingCapacity> input_ids{};
  std::array<std::uint64_t, BatchMapBindingCapacity> output_ids{};
  std::size_t input_count = 0u;
  std::size_t output_count = 0u;
  std::uint64_t max_tiles = 0u;
  bool eligible = false;
};

struct BatchMapGroup final {
  std::size_t begin = 0u;
  std::size_t count = 0u;
  std::array<std::uint64_t, BatchMapBindingCapacity> inputs{};
  std::array<std::uint64_t, BatchMapBindingCapacity> outputs{};
  bool packed = false;
};

struct BatchMapPlan final {
  std::array<BatchMapGroup, BatchMapCapacity> groups{};
  std::size_t size = 0u;
  std::uint64_t input_bytes = 0u;
  std::uint64_t output_bytes = 0u;
  bool ok = false;
  const char *reason = "compute_batch_layout_invalid";
};

[[nodiscard]] constexpr bool SameBatchMap(const BatchMapView &left,
                                          const BatchMapView &right) noexcept {
  if (!left.eligible || !right.eligible || left.kernel != right.kernel ||
      left.tiles != right.tiles || left.input_count != right.input_count ||
      left.output_count != right.output_count ||
      left.max_tiles != right.max_tiles) {
    return false;
  }
  for (std::size_t index = 0u; index < left.input_count; ++index) {
    if (left.inputs[index] != right.inputs[index]) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < left.output_count; ++index) {
    if (left.outputs[index] != right.outputs[index]) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool
BatchMapAliasFree(const std::span<const BatchMapView> views,
                  const std::size_t begin, const std::size_t end) noexcept {
  constexpr std::size_t IdCapacity = BatchMapCapacity * BatchMapBindingCapacity;
  std::array<std::uint64_t, IdCapacity> inputs{};
  std::array<std::uint64_t, IdCapacity> outputs{};
  std::size_t input_count = 0u;
  std::size_t output_count = 0u;
  for (std::size_t job = begin; job < end; ++job) {
    const BatchMapView &view = views[job];
    if (view.input_count > BatchMapBindingCapacity ||
        view.output_count > BatchMapBindingCapacity ||
        view.input_count > inputs.size() - input_count ||
        view.output_count > outputs.size() - output_count) {
      return false;
    }
    for (std::size_t input = 0u; input < view.input_count; ++input) {
      inputs[input_count++] = view.input_ids[input];
    }
    for (std::size_t output = 0u; output < view.output_count; ++output) {
      const std::uint64_t id = view.output_ids[output];
      if (id == 0u) {
        return false;
      }
      outputs[output_count++] = id;
    }
  }
  std::sort(inputs.begin(), inputs.begin() + input_count);
  std::sort(outputs.begin(), outputs.begin() + output_count);
  if (std::adjacent_find(outputs.begin(), outputs.begin() + output_count) !=
      outputs.begin() + output_count) {
    return false;
  }
  std::size_t input = 0u;
  std::size_t output = 0u;
  while (input < input_count && output < output_count) {
    if (inputs[input] < outputs[output]) {
      ++input;
    } else if (outputs[output] < inputs[input]) {
      ++output;
    } else {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool AlignBatchBytes(const std::uint64_t value,
                                             const std::uint64_t alignment,
                                             std::uint64_t &out) noexcept {
  if (alignment == 0u) {
    return false;
  }
  const std::uint64_t remainder = value % alignment;
  const std::uint64_t padding = remainder == 0u ? 0u : alignment - remainder;
  if (!rund::kernel::checked::add(value, padding)) {
    return false;
  }
  out = value + padding;
  return true;
}

[[nodiscard]] inline BatchMapPlan
PlanBatchMaps(const std::span<const BatchMapView> views,
              const std::uint64_t alignment,
              const std::uint64_t workspace_limit) noexcept {
  BatchMapPlan plan{};
  if (views.empty() || views.size() > plan.groups.size() || alignment == 0u ||
      workspace_limit == 0u) {
    return plan;
  }

  std::size_t begin = 0u;
  while (begin < views.size()) {
    std::size_t end = begin + 1u;
    while (end < views.size() && SameBatchMap(views[begin], views[end])) {
      ++end;
    }
    BatchMapGroup group{.begin = begin, .count = end - begin};
    if (group.count > 1u && views[begin].eligible &&
        BatchMapAliasFree(views, begin, end)) {
      const BatchMapView &view = views[begin];
      std::uint64_t input_cursor = plan.input_bytes;
      std::uint64_t output_cursor = plan.output_bytes;
      bool valid = rund::kernel::checked::mul(view.tiles, group.count) &&
                   view.tiles * group.count <= view.max_tiles;
      for (std::size_t index = 0u; index < view.input_count; ++index) {
        std::uint64_t offset = 0u;
        if (!AlignBatchBytes(input_cursor, alignment, offset) ||
            !rund::kernel::checked::mul(view.tiles, view.inputs[index]) ||
            !rund::kernel::checked::mul(view.tiles * view.inputs[index],
                                        group.count)) {
          valid = false;
          break;
        }
        const std::uint64_t bytes =
            view.tiles * view.inputs[index] * group.count;
        if (!rund::kernel::checked::add(offset, bytes)) {
          valid = false;
          break;
        }
        group.inputs[index] = offset;
        input_cursor = offset + bytes;
      }
      for (std::size_t index = 0u; valid && index < view.output_count;
           ++index) {
        std::uint64_t offset = 0u;
        if (!AlignBatchBytes(output_cursor, alignment, offset) ||
            !rund::kernel::checked::mul(view.tiles, view.outputs[index]) ||
            !rund::kernel::checked::mul(view.tiles * view.outputs[index],
                                        group.count)) {
          valid = false;
          break;
        }
        const std::uint64_t bytes =
            view.tiles * view.outputs[index] * group.count;
        if (!rund::kernel::checked::add(offset, bytes)) {
          valid = false;
          break;
        }
        group.outputs[index] = offset;
        output_cursor = offset + bytes;
      }
      if (!valid) {
        return plan;
      }
      const bool within_limit =
          input_cursor <= workspace_limit && output_cursor <= workspace_limit &&
          rund::kernel::checked::add(input_cursor, output_cursor) &&
          input_cursor + output_cursor <= workspace_limit;
      if (within_limit) {
        group.packed = true;
        plan.input_bytes = input_cursor;
        plan.output_bytes = output_cursor;
      }
    }
    plan.groups[plan.size++] = group;
    begin = end;
  }
  plan.ok = true;
  plan.reason = "ok";
  return plan;
}

} // namespace rund::node::accel::detail
