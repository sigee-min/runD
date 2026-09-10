#pragma once

#include "../scratch.hpp"

#include <array>

namespace rund::node::accel::detail {

class Operation;

namespace kernel_scratch_internal {

struct ScratchRequests final {
  std::array<std::uint64_t, 8u> bytes{};
  std::size_t count{};
  bool ok{true};
};

[[nodiscard]] ScratchRequests BuildScratchRequests(const Operation &operation,
                                                   rund::AccelApi api) noexcept;

} // namespace kernel_scratch_internal

} // namespace rund::node::accel::detail
