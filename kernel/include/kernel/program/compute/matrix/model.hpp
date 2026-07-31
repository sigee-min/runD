#pragma once

#include <kernel/core/model.hpp>
#include <kernel/program/compute/model.hpp>

namespace rund::kernel {

enum class MatrixOp : u8 {
  Mul = 1u,
  Transpose = 2u,
  BatchMul = 3u,
};

enum class MatrixLayout : u8 {
  RowMajor = 1u,
  ColumnMajor = 2u,
};

enum class MatrixArithmetic : u8 {
  Fixed = 1u,
  SignedWrap = 2u,
  UnsignedWrap = 3u,
};

struct MatrixShape {
  MatrixLayout layout = MatrixLayout::RowMajor;
  MatrixArithmetic arithmetic = MatrixArithmetic::Fixed;
  u64 rows = 0u;
  u64 cols = 0u;
  u64 inner = 0u;
  u64 batch_count = 1u;
  u32 element_bytes = 4u;
};

struct MatrixDesc {
  MatrixOp op = MatrixOp::Mul;
  MatrixLayout layout = MatrixLayout::RowMajor;
  MatrixArithmetic arithmetic = MatrixArithmetic::Fixed;
  u64 rows = 0u;
  u64 cols = 0u;
  u64 inner = 0u;
  u64 batch_count = 1u;
  u32 element_bytes = 4u;
  ComputeFixedFormat fixed_format{};
};

struct MatrixPlan {
  MatrixOp op = MatrixOp::Mul;
  MatrixLayout layout = MatrixLayout::RowMajor;
  MatrixArithmetic arithmetic = MatrixArithmetic::Fixed;
  u64 rows = 0u;
  u64 cols = 0u;
  u64 inner = 0u;
  u64 batch_count = 0u;
  u64 left_count = 0u;
  u64 right_count = 0u;
  u64 output_count = 0u;
  u32 element_bytes = 0u;
  ComputeFixedFormat fixed_format{};
  u64 pass_count = 0u;
  bool ok = false;
  const char* reason = "compute_matrix_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

struct MatrixHash {
  u64 hi = 0u;
  u64 lo = 0u;
};

struct MatrixResult {
  u64 output_count = 0u;
  bool ok = false;
  const char* reason = "compute_matrix_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok;
  }
};

}  // namespace rund::kernel
