#pragma once

#include <kernel/program/compute/lowering/names.hpp>

namespace rund::kernel::compute_lowering_detail {

[[nodiscard]] ComputeDomain
BindingDomainForShape(const ParsedBinding &binding) noexcept;

[[nodiscard]] const char *ValidateLowerableIR(const ParsedIR &parsed,
                                              ComputeScalar scalar);

} // namespace rund::kernel::compute_lowering_detail
