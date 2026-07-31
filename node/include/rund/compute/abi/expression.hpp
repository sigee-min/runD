#pragma once

#include <rund/compute/abi/model.hpp>
namespace rund::compute::detail {
[[nodiscard]] std::shared_ptr<ExprState> make_expr();
[[nodiscard]] ExprRef input(const std::shared_ptr<ExprState> &state, Type type,
                            std::uint32_t index, FixedFormat fixed_format = {});
[[nodiscard]] ExprRef constant(const std::shared_ptr<ExprState> &state,
                               Type type, std::uint64_t bits,
                               FixedFormat fixed_format = {});
[[nodiscard]] ExprRef index(const std::shared_ptr<ExprState> &state, Type type);
[[nodiscard]] ExprRef unary(ExprOp operation, ExprRef value);
[[nodiscard]] ExprRef shift(ExprOp operation, ExprRef value,
                            std::uint32_t amount);
[[nodiscard]] ExprRef retype_expr(ExprRef value, Type type);
[[nodiscard]] ExprRef checked_ordinal_expr(ExprRef value, Type type);
[[nodiscard]] ExprRef boundary_mask_expr(ExprRef value, Type type,
                                         FixedFormat fixed_format);
[[nodiscard]] ExprRef with_fixed_format(ExprRef value,
                                        FixedFormat fixed_format);
[[nodiscard]] ExprRef quantize_expr(ExprRef value, Type target,
                                    FixedFormat fixed_format);
[[nodiscard]] ExprRef make_mask(ExprRef predicate);
[[nodiscard]] ExprRef make_mask(ExprRef predicate, Type output);
[[nodiscard]] bool is_width_mask(const ExprRef &expression,
                                 Type input) noexcept;
[[nodiscard]] ExprRef binary(ExprOp operation, ExprRef left, ExprRef right);
[[nodiscard]] ExprRef ternary(ExprOp operation, ExprRef first, ExprRef second,
                              ExprRef third);
} // namespace rund::compute::detail
