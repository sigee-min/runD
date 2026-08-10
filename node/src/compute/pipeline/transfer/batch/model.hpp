#pragma once

#include "../../state.hpp"

#include <cstddef>
#include <cstdint>

namespace rund::compute::detail {

[[nodiscard]] Status
validate_pipeline_transfer_ready(const PipelineState &state) noexcept;

[[nodiscard]] bool add_pipeline_transfer_bytes(std::size_t bytes,
                                               std::uint64_t &total) noexcept;

} // namespace rund::compute::detail
