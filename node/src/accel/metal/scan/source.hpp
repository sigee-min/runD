#pragma once

#include <string>

namespace rund::node::accel::detail {

[[nodiscard]] std::string MetalScanSource();
[[nodiscard]] bool MetalScanSourceUpperBytes(std::uint64_t &upper) noexcept;

}  // namespace rund::node::accel::detail
