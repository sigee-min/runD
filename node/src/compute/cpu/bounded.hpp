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
  const kernel::ComputeTilePrepareResult prepared = run.tiles.prepare(count);
  if (!prepared.ok || prepared.tile_count > run.totals.capacity() ||
      (run.needs_prefixes && prepared.tile_count > run.prefixes.capacity())) {
    return Status::fail(
        prepared.ok
            ? Reason::TileRunCapacity
            : project_reason(prepared.reason, Reason::TileBackendFailed));
  }
  run.totals.resize(prepared.tile_count);
  if (run.needs_prefixes) {
    run.prefixes.resize(prepared.tile_count);
  }
  run.tile_size = prepared.tile_units;
  return Status::success();
}

} // namespace rund::compute::detail
