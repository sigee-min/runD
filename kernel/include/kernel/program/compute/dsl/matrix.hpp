#pragma once

namespace rund::compute_dsl {

struct MatOpTranspose final {};
struct MatOpSolve final {};
struct MatOpDeterminant final {};
struct MatOpTrace final {};

struct MatOp final {
  inline static constexpr MatOpTranspose Transpose{};
  inline static constexpr MatOpSolve Solve{};
  inline static constexpr MatOpDeterminant Determinant{};
  inline static constexpr MatOpTrace Trace{};
};

} // namespace rund::compute_dsl
