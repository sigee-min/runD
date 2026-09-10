#pragma once

#include "../state.hpp"

namespace rund::compute::detail::virtual_observe {

[[nodiscard]] bool valid_input_authority(const VirtualPipelineState &,
                                         std::size_t minimum,
                                         std::size_t maximum) noexcept;
[[nodiscard]] bool
valid_poolless_device_vsm(const VirtualPipelineState &) noexcept;
[[nodiscard]] bool valid_graph(const VirtualPipelineState &) noexcept;
[[nodiscard]] MemoryStats
virtual_memory_locked(const VirtualPipelineState &) noexcept;

} // namespace rund::compute::detail::virtual_observe
