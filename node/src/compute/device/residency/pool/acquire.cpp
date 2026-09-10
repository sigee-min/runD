#include "../pool.hpp"

#include "acquire/common.hpp"
#include "internal.hpp"

#include "../../../pipeline/plan/contract.hpp"
#include "../../../pipeline/residency/model.hpp"
#include "../../../size.hpp"
#include "../../../type.hpp"
#include "../../state.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <utility>

namespace rund::compute::detail::residency {

std::shared_ptr<Pool> Registry::acquire(
    const std::shared_ptr<DeviceState> &device, const PoolLayout &layout,
    const std::span<const TiledGraphPhysicalClass> graph_classes) noexcept {
  if (device == nullptr || layout.input_page_bytes == 0u ||
      layout.output_page_bytes == 0u || layout.frame_capacity == 0u ||
      layout.graph_host_input_count == 0u || layout.host_frame_capacity == 0u ||
      layout.host_output_frame_capacity == 0u ||
      layout.host_frame_capacity < layout.frame_capacity ||
      layout.host_output_frame_capacity < layout.frame_capacity ||
      layout.frame_capacity >
          std::numeric_limits<std::uint32_t>::max() / Pool::BankCount ||
      layout.host_frame_capacity >
          std::numeric_limits<std::uint32_t>::max() /
              layout.graph_host_input_count ||
      layout.host_frame_capacity * layout.graph_host_input_count >
          std::numeric_limits<std::uint32_t>::max() / Pool::BankCount ||
      layout.host_output_frame_capacity >
          std::numeric_limits<std::uint32_t>::max() / Pool::BankCount) {
    return nullptr;
  }
  if (graph_classes.size() > TiledGraphResourceCapacity) {
    return nullptr;
  }

  acquire_detail::RegistryKeepalive keepalive;
  for (std::size_t index = 0u; index < graph_classes.size(); ++index) {
    const TiledGraphPhysicalClass &physical = graph_classes[index];
    if (physical.physical_id != index + 1u || !valid_type(physical.type) ||
        !valid_format(physical.type, physical.format) ||
        physical.page_bytes == 0u ||
        physical.page_bytes % type_bytes(physical.type) != 0u) {
      return nullptr;
    }
  }
  if (!graph_classes.empty() && device->backend != Backend::Cpu) {
    const std::size_t input_count = static_cast<std::size_t>(
        std::count_if(graph_classes.begin(), graph_classes.end(),
                      [](const TiledGraphPhysicalClass &physical) {
                        return physical.role == GraphResourceRole::Input;
                      }));
    const std::size_t output_count = static_cast<std::size_t>(
        std::count_if(graph_classes.begin(), graph_classes.end(),
                      [](const TiledGraphPhysicalClass &physical) {
                        return physical.role == GraphResourceRole::Output;
                      }));
    if ((layout.graph_host_service &&
         (input_count != layout.graph_host_input_count ||
          input_count >= TiledGraphResourceCapacity)) ||
        input_count == 0u || output_count != 1u) {
      return nullptr;
    }
  }
  const std::size_t input_width = type_bytes(layout.input_type);
  const std::size_t intermediate_width = type_bytes(layout.intermediate_type);
  const std::size_t control_width = type_bytes(layout.control_type);
  const std::size_t output_width = type_bytes(layout.output_type);
  if (input_width == 0u || output_width == 0u ||
      layout.input_page_bytes % input_width != 0u ||
      layout.output_page_bytes % output_width != 0u ||
      (layout.intermediate_page_bytes != 0u &&
       (intermediate_width == 0u ||
        layout.intermediate_page_bytes % intermediate_width != 0u)) ||
      (layout.control_page_bytes != 0u &&
       (control_width == 0u ||
        layout.control_page_bytes % control_width != 0u))) {
    return nullptr;
  }

  // The gate remains the sole public admission boundary. Projection TUs
  // below may allocate and register, but cannot run concurrently with a
  // second Registry acquisition or release.
  std::lock_guard lock{gate_};
  try {
    keepalive.pools.reserve(pools_.size());
    keepalive.arenas.reserve(arenas_.size());
    for (auto cursor = pools_.begin(); cursor != pools_.end();) {
      if (auto pool = cursor->lock()) {
        keepalive.pools.push_back(std::move(pool));
        ++cursor;
      } else {
        cursor = pools_.erase(cursor);
      }
    }
    for (auto cursor = arenas_.begin(); cursor != arenas_.end();) {
      if (auto arena = cursor->lock()) {
        keepalive.arenas.push_back(std::move(arena));
        ++cursor;
      } else {
        cursor = arenas_.erase(cursor);
      }
    }
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
  for (const std::shared_ptr<Pool> &pool : keepalive.pools) {
    if (pool->layout == layout && graph_classes_match(*pool, graph_classes)) {
      return pool;
    }
  }
  return graph_classes.empty()
             ? acquire_detail::acquire_ordinary(*this, device, layout,
                                                keepalive)
             : acquire_detail::acquire_graph(*this, device, layout,
                                             graph_classes, keepalive);
}

} // namespace rund::compute::detail::residency
