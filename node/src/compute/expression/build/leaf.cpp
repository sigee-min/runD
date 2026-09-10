#include "internal.hpp"

#include <memory>

namespace rund::compute::detail {

std::shared_ptr<ExprState> make_expr() {
  try {
    return std::make_shared<ExprState>();
  } catch (const std::bad_alloc &) {
    return {};
  }
}

ExprRef input(const std::shared_ptr<ExprState> &state, const Type type,
              const std::uint32_t index, const FixedFormat fixed_format) {
  return expression_build::append(state, ExprNode{
                                             .operation = ExprOp::Input,
                                             .type = type,
                                             .fixed_format = fixed_format,
                                             .left = index,
                                         });
}

ExprRef constant(const std::shared_ptr<ExprState> &state, const Type type,
                 const std::uint64_t bits, const FixedFormat fixed_format) {
  return expression_build::append(state, ExprNode{
                                             .operation = ExprOp::Constant,
                                             .type = type,
                                             .fixed_format = fixed_format,
                                             .bits = bits,
                                         });
}

ExprRef index(const std::shared_ptr<ExprState> &state, const Type type) {
  return expression_build::append(
      state, ExprNode{.operation = ExprOp::Index, .type = type});
}

} // namespace rund::compute::detail
