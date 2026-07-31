#pragma once

#include "model.hpp"

#include <rund/compute/status.hpp>

namespace rund::compute::detail {

[[nodiscard]] Status run_stencil(PrimitiveContext &context);
[[nodiscard]] Status run_transform(PrimitiveContext &context);
[[nodiscard]] Status run_matrix(PrimitiveContext &context);
[[nodiscard]] Status run_factor(PrimitiveContext &context);
[[nodiscard]] Status run_solve(PrimitiveContext &context);
[[nodiscard]] Status run_spectrum(PrimitiveContext &context);

} // namespace rund::compute::detail
