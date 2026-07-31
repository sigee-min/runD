#pragma once

#include <string>
#include <string_view>

namespace rund::node::accel::detail {

[[nodiscard]] std::string MetalPipelineSource();
[[nodiscard]] std::string_view MetalPipelineStatusSource() noexcept;
[[nodiscard]] std::string_view MetalPipelineTelemetrySourceText() noexcept;

} // namespace rund::node::accel::detail
