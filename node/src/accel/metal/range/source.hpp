#pragma once

#include <cstdint>
#include <string>

namespace rund::node::accel::detail {
class RangeExec;
class RangePlan;
[[nodiscard]] std::string MetalRangeSource(const RangeExec &execution);
[[nodiscard]] bool MetalRangeSourceUpperBytes(const RangeExec &execution,
                                              std::uint64_t &upper) noexcept;
[[nodiscard]] std::string MetalRangeControlSource(const RangePlan &plan);
[[nodiscard]] bool
MetalRangeControlSourceUpperBytes(const RangePlan &plan,
                                  std::uint64_t &upper) noexcept;
} // namespace rund::node::accel::detail
