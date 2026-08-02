#pragma once

#include <rund/compute/status.hpp>

#include <string_view>

namespace rund::compute::detail {

[[nodiscard]] std::string_view reason_message(Reason reason) noexcept;
[[nodiscard]] Reason project_reason(std::string_view reason,
                                    Reason boundary) noexcept;
[[nodiscard]] Reason
project_pipeline_preparation_reason(std::string_view reason) noexcept;

} // namespace rund::compute::detail
