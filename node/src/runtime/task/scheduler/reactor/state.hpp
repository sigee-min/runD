#pragma once

#include <cstdint>

namespace rund::node {

[[nodiscard]] std::uint64_t ReactorHostHandleId(
    int fd,
    std::uint64_t explicit_host_handle_id) noexcept;

}  // namespace rund::node
