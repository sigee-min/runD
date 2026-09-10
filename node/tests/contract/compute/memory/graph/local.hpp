#pragma once

#include <rund/compute.hpp>

#include <array>
#include <cstdint>
#include <memory>

namespace rund::compute::detail {
struct ProgramState;
}

namespace rund_node_memory_contract::graph_detail {

inline constexpr std::array<std::uint32_t, 4u> FirstInput{1u, 2u, 3u, 4u};
inline constexpr std::array<std::uint32_t, 4u> SecondInput{4u, 3u, 2u, 1u};

[[nodiscard]] int
CheckResidentJobMemory(rund::compute::Program<std::uint32_t(std::uint32_t)> &,
                       rund::compute::Device &, std::uint64_t baseline);
[[nodiscard]] int CheckViewRequirements(
    const std::shared_ptr<rund::compute::detail::ProgramState> &);

} // namespace rund_node_memory_contract::graph_detail
