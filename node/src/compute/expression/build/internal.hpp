#pragma once

#include "../state.hpp"

#include <memory>

namespace rund::compute::detail::expression_build {

void set_error(const std::shared_ptr<ExprState> &state, Status status);

[[nodiscard]] bool stored_unary(ExprOp operation) noexcept;
[[nodiscard]] bool stored_binary(ExprOp operation) noexcept;
[[nodiscard]] bool approximate_unary(ExprOp operation) noexcept;
[[nodiscard]] bool approximate_binary(ExprOp operation) noexcept;
[[nodiscard]] bool stored_format(Type type, FixedFormat format) noexcept;

[[nodiscard]] ExprRef append(const std::shared_ptr<ExprState> &state,
                             ExprNode node);
[[nodiscard]] bool valid(const ExprRef &value) noexcept;

} // namespace rund::compute::detail::expression_build
