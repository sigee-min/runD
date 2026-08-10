#pragma once

#include "projection.hpp"

#include <rund/compute/stats.hpp>

namespace rund::compute::detail {

[[nodiscard]] Status
validate_virtual_recovery(const VirtualBacking &input,
                          const VirtualBacking &output,
                          std::uint64_t required_output_bytes) noexcept;

[[nodiscard]] Status read_virtual_wave(VirtualBacking &backing,
                                       const VirtualWaveProjection &wave,
                                       const VirtualRunProjection &run,
                                       ResidencyStats &stats) noexcept;

[[nodiscard]] Status write_virtual_wave(VirtualBacking &backing,
                                        const VirtualWaveProjection &wave,
                                        const VirtualRunProjection &run,
                                        ResidencyStats &stats) noexcept;

void clear_virtual_recovery(VirtualBacking &backing) noexcept;

} // namespace rund::compute::detail
