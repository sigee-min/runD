#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <rund/compute/abi/state.hpp>
#include <rund/compute/ops.hpp>

namespace rund::compute::detail {
[[nodiscard]] std::optional<std::uint32_t>
fold_filter_sum(const std::shared_ptr<FlowState> &, std::uint32_t values,
                std::uint32_t count, Reduce operation);
} // namespace rund::compute::detail
