#pragma once

#include <string_view>

namespace rund::node::accel::detail::metal_pipeline_status_source {

[[nodiscard]] std::string_view preamble() noexcept;
[[nodiscard]] std::string_view abi() noexcept;
[[nodiscard]] std::string_view reset() noexcept;
[[nodiscard]] std::string_view publish() noexcept;
[[nodiscard]] std::string_view advance() noexcept;
[[nodiscard]] std::string_view reduce() noexcept;

} // namespace rund::node::accel::detail::metal_pipeline_status_source
