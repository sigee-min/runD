#pragma once

#include "../../graph/state.hpp"

#include <rund/compute/abi/model.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

namespace rund::compute::detail {

[[nodiscard]] ValueIds graph_map_indexed(
    const std::shared_ptr<GraphState> &graph,
    std::span<const std::uint32_t> sources,
    std::span<const std::uint32_t> indices,
    std::span<const ExprRef> expressions, FlowControl control,
    std::string_view name, std::size_t output_count);

} // namespace rund::compute::detail
