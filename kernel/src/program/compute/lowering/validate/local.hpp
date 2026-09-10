#pragma once

#include "model.hpp"

namespace rund::kernel::compute_lowering_detail {

[[nodiscard]] bool CanonicalMaskConstant(const ParsedIR &parsed, u32 node_ref,
                                         u32 expected_low_bits) noexcept;
[[nodiscard]] bool CanonicalMaskSource(const ParsedIR &parsed, u32 source_ref,
                                       bool fixed_mode) noexcept;
[[nodiscard]] bool CanonicalUnsignedMaskWriteBinding(const ParsedIR &parsed,
                                                     u32 binding) noexcept;

} // namespace rund::kernel::compute_lowering_detail
