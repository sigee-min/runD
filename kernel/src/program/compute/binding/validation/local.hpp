#pragma once

#include <kernel/program/compute/binding/validation.hpp>

namespace rund::kernel::binding_validation_detail {

[[nodiscard]] BindingValidation Reject(const char *) noexcept;
[[nodiscard]] BindingValidation Accept() noexcept;
[[nodiscard]] bool SpanCanAddressTiles(const BufferSpan &, u64) noexcept;
[[nodiscard]] bool OutputCanAddressTiles(const BindingSet &) noexcept;
[[nodiscard]] bool OutputCanAddressTiles(const OutputSpan &, u64) noexcept;
[[nodiscard]] BindingValidation
ExpectedInputElementBytes(const BindingObligations &, u64, u64 &) noexcept;
[[nodiscard]] BindingValidation ExpectedInputCount(const BindingObligations &,
                                                   u64, u64 &) noexcept;
[[nodiscard]] BindingValidation
ExpectedOutputElementBytes(const BindingObligations &, u64, u64 &) noexcept;
[[nodiscard]] BindingValidation
ValidateInputs(const BindingSet &, const BindingObligations &) noexcept;
[[nodiscard]] BindingValidation
ValidateOutputs(const BindingSet &, const BindingObligations &) noexcept;

} // namespace rund::kernel::binding_validation_detail
