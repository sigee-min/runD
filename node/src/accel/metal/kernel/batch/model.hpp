#pragma once

#include "../../../kernel/batch/plan.hpp"
#include "../../runtime/map/resources.hpp"
#include "../../state.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::node::accel::detail::metalbatch {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct EntryKey final {
  const void *steps = nullptr;
  const void *artifact = nullptr;
  std::uint64_t kernel = 0u;
  std::uint64_t tiles = 0u;
  std::size_t step_count = 0u;
  std::uint64_t input_count = 0u;
  std::uint64_t output_count = 0u;
  bool requires_reset = false;
};

struct Workspace final {
  MetalAdapter *adapter = nullptr;
  MetalRuntimeBuffer input{};
  MetalRuntimeBuffer output{};
  std::array<EntryKey, BatchMapCapacity> entries{};
  std::array<BatchMapView, BatchMapCapacity> views{};
  BatchMapPlan plan{};
  std::size_t size = 0u;
  bool planned = false;

  ~Workspace();
};

using Maps = std::array<MetalMapEncodeResources *, BatchMapCapacity>;
#endif

} // namespace rund::node::accel::detail::metalbatch
