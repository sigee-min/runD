#pragma once

#include <rund/compute/status.hpp>

#include <string_view>

namespace rund::compute::detail {

[[nodiscard]] std::string_view reason_message(Reason reason) noexcept;
[[nodiscard]] Reason project_reason(std::string_view reason,
                                    Reason boundary) noexcept;

} // namespace rund::compute::detail
