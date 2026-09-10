#pragma once

#include <string_view>

namespace rund::node::accel::detail {

[[nodiscard]] std::string_view MetalNestedAggregatePreambleSource() noexcept;
[[nodiscard]] std::string_view MetalNestedAggregateCommonSource() noexcept;

} // namespace rund::node::accel::detail
