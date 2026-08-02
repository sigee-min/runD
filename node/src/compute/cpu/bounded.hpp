#pragma once

#include "../status.hpp"
#include "../type.hpp"
#include "state.hpp"
#include "view.hpp"

#include <cstring>
#include <limits>

namespace rund::compute::detail {

template <class View>
[[nodiscard]] inline Status read_control_value(const BufferState *const buffer,
                                               const View view, const Type type,
                                               std::uint64_t &value) noexcept {
  if ((type != Type::U32 && type != Type::U64) ||
      view.element_bytes != type_bytes(type)) {
    return Status::fail(Reason::BoundedCountInvalid);
  }
  const std::optional<CpuView> source =
      cpu_view(buffer, view.offset, 1u, 1u, view.element_bytes);
  if (!source || source->data == nullptr) {
    return Status::fail(Reason::BoundedCountInvalid);
  }
  value = 0u;
  if (view.element_bytes == sizeof(std::uint32_t)) {
    std::uint32_t narrow = 0u;
    std::memcpy(&narrow, source->data, sizeof(narrow));
    value = narrow;
  } else {
    std::memcpy(&value, source->data, sizeof(value));
  }
  return Status::success();
}

template <class View>
[[nodiscard]] inline Status read_bounded_count(const BufferState *const buffer,
                                               const View view, const Type type,
                                               const std::size_t capacity,
                                               kernel::u32 &count) noexcept {
  std::uint64_t logical = 0u;
  const Status read = read_control_value(buffer, view, type, logical);
  if (!read) {
    return read;
  }
  if (logical > capacity || logical > std::numeric_limits<kernel::u32>::max()) {
    return Status::fail(Reason::BoundedCountInvalid);
  }
  count = static_cast<kernel::u32>(logical);
  return Status::success();
}

[[nodiscard]] inline Status
prepare_bounded_collective(CpuCollectiveRun &run,
                           const kernel::u32 count) noexcept {
  if (!run.tile_plan.prepared() || run.execution == nullptr) {
    return Status::fail(Reason::TileRunCapacity);
  }
  run.tiles = run.tile_plan.bind(run.execution->tile_storage(), count);
  const std::size_t tile_count = run.tiles.tile_count();
  if (!run.tiles.prepared() || !run.tiles.has_run_storage() ||
      run.tiles.count() != count || tile_count > run.total_capacity.size() ||
      (run.needs_prefixes && tile_count > run.prefix_capacity.size())) {
    return Status::fail(Reason::TileRunCapacity);
  }
  run.totals = run.total_capacity.first(tile_count);
  run.prefixes = run.needs_prefixes
                     ? run.prefix_capacity.first(tile_count)
                     : std::span<CpuCollectiveWide>{};
  run.tile_size = tile_count == 0u
                      ? 0u
                      : (static_cast<std::uint64_t>(count) + tile_count - 1u) /
                            tile_count;
  return Status::success();
}

} // namespace rund::compute::detail
