#pragma once

#include <kernel/program/compute/binding/model.hpp>

namespace rund::kernel {
[[nodiscard]] BindingValidation
ValidateResidentBuffer(const ResidentBufferRef &ref, u64 tile_count,
                       u64 expected_element_bytes, u32 expected_usage,
                       bool allow_stride = false) noexcept;

[[nodiscard]] BindingValidation
ValidateRuntimeBindings(const BindingSet &bindings,
                        const BindingObligations &obligations) noexcept;

} // namespace rund::kernel
