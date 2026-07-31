#pragma once

#include "state.hpp"

#include <memory>
#include <span>

namespace rund::compute::detail {

struct DeviceState;
struct GraphState;

[[nodiscard]] Result<std::shared_ptr<GraphState>>
materialize_graph(const std::shared_ptr<FlowState> &flow,
                  const std::shared_ptr<DeviceState> &device,
                  std::span<const std::size_t> order,
                  std::span<const MapRecipe> maps,
                  std::span<const std::uint8_t> skipped = {});

} // namespace rund::compute::detail
