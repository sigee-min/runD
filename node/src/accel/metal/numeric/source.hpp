#pragma once

#include "../../kernel/backend/source_recipe.hpp"

#include <cstdint>
#include <string>

namespace rund::node::accel::detail {

[[nodiscard]] bool EmitMetalNumericFixedLane32Source(
    backend_source_recipe::CountSink &sink) noexcept;
[[nodiscard]] bool EmitMetalNumericFixedLane32Source(
    backend_source_recipe::StringSink &sink);
[[nodiscard]] bool EmitMetalNumericFixedLane64Source(
    backend_source_recipe::CountSink &sink) noexcept;
[[nodiscard]] bool EmitMetalNumericFixedLane64Source(
    backend_source_recipe::StringSink &sink);
[[nodiscard]] std::string MetalNumericSource();
[[nodiscard]] bool MetalNumericSourceUpperBytes(
    std::uint64_t &upper) noexcept;

}  // namespace rund::node::accel::detail
