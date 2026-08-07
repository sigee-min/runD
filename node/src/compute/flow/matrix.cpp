#include "recipe.hpp"

#include "../size.hpp"
#include "../type.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <span>

namespace rund::compute::detail {

namespace {

[[nodiscard]] bool matrix_count(FlowState &flow, const std::size_t rows,
                                const std::size_t cols,
                                const std::size_t batches,
                                std::size_t &count) noexcept {
  if (rows == 0u || cols == 0u || batches == 0u) {
    reject(flow, Reason::MatrixShapeZero);
    return false;
  }
  if (!size::multiply(rows, cols, batches, count)) {
    reject(flow, Reason::MatrixShapeOverflow);
    return false;
  }
  return true;
}

} // namespace

void flow_matrix_view(const std::shared_ptr<FlowState> &flow,
                      const std::size_t rows, const std::size_t cols,
                      const std::size_t batches) {
  if (flow == nullptr || !flow->status) {
    return;
  }
  std::size_t count = 0u;
  if (matrix_count(*flow, rows, cols, batches, count) &&
      count != flow_output_count(flow)) {
    reject(*flow, Reason::MatrixShapeMismatch);
  }
}

std::size_t flow_matrix_extent(const std::shared_ptr<FlowState> &flow,
                               const std::size_t rows, const std::size_t cols,
                               const std::size_t batches) {
  if (flow == nullptr || !flow->status) {
    return 0u;
  }
  std::size_t count = 0u;
  (void)matrix_count(*flow, rows, cols, batches, count);
  return count;
}

std::size_t flow_matrix_product(
    const std::shared_ptr<FlowState> &flow, const std::size_t left_rows,
    const std::size_t left_cols, const std::size_t left_batches,
    const std::size_t right_rows, const std::size_t right_cols,
    const std::size_t right_batches, const std::size_t right_count) {
  if (flow == nullptr || !flow->status) {
    return 0u;
  }
  if (left_cols != right_rows || left_batches != right_batches) {
    reject(*flow, Reason::MatrixShapeMismatch);
    return 0u;
  }
  std::size_t expected_right = 0u;
  if (!matrix_count(*flow, right_rows, right_cols, right_batches,
                    expected_right)) {
    return 0u;
  }
  if (expected_right != right_count) {
    reject(*flow, Reason::MatrixShapeMismatch);
    return 0u;
  }
  std::size_t output = 0u;
  (void)matrix_count(*flow, left_rows, right_cols, left_batches, output);
  return output;
}

FactorIds flow_factor(const std::shared_ptr<FlowState> &flow,
                      const FactorOp operation, const std::size_t rows,
                      const std::size_t cols, const std::size_t batches) {
  if (flow == nullptr || !flow->status) {
    return {};
  }
  std::size_t matrix_size = 0u;
  if (!matrix_count(*flow, rows, cols, batches, matrix_size)) {
    return {};
  }
  if (matrix_size != flow_output_count(flow)) {
    reject(*flow, Reason::FactorShapeMismatch);
    return {};
  }
  if ((operation == FactorOp::Lu || operation == FactorOp::Cholesky) &&
      rows != cols) {
    reject(*flow, Reason::FactorShapeSquare);
    return {};
  }
  const FlowValue value = flow->values[flow->output - 1u];
  std::size_t factor_size = matrix_size;
  if (operation == FactorOp::Qr &&
      !size::multiply(matrix_size, 2u, factor_size)) {
    reject(*flow, Reason::FactorShapeOverflow);
    return {};
  }
  FixedFormat fixed_format = value.fixed_format;
  if (type_fixed(value.type)) {
    fixed_format.approximation = Approximation::Deterministic;
  }
  const std::uint32_t packed =
      append(*flow, value.type, factor_size, fixed_format);
  const std::uint32_t pivots =
      operation == FactorOp::Lu ? append(*flow, Type::U32, rows * batches) : 0u;
  const std::uint32_t status = append(*flow, Type::U32, batches);
  if (packed == 0u || status == 0u ||
      (operation == FactorOp::Lu && pivots == 0u)) {
    return {};
  }
  const std::array inputs{flow->output};
  const std::array lu_outputs{packed, pivots, status};
  const std::array direct_outputs{packed, status};
  const std::span<const std::uint32_t> stored_outputs =
      operation == FactorOp::Lu
          ? std::span<const std::uint32_t>{lu_outputs}
          : std::span<const std::uint32_t>{direct_outputs};
  if (!append_primitive(*flow, inputs, stored_outputs, Primitive::Factor,
                        {.first = rows,
                         .second = cols,
                         .third = batches,
                         .mode = static_cast<std::uint32_t>(operation)})) {
    return {};
  }
  flow->output = packed;
  return FactorIds{packed, pivots, status};
}

SolveIds flow_factor_solve(const std::shared_ptr<FlowState> &flow,
                           const FactorOp operation, const std::uint32_t packed,
                           const std::uint32_t pivots, const std::uint32_t rhs,
                           const std::size_t rows, const std::size_t rhs_cols,
                           const std::size_t batches) {
  if (flow == nullptr || !flow->status || packed == 0u || rhs == 0u ||
      packed > flow->values.size() || rhs > flow->values.size() ||
      (operation == FactorOp::Lu &&
       (pivots == 0u || pivots > flow->values.size()))) {
    return {};
  }
  std::size_t rhs_count = 0u;
  if (!matrix_count(*flow, rows, rhs_cols, batches, rhs_count)) {
    return {};
  }
  const FlowValue &factor = flow->values[packed - 1u];
  const FlowValue &right = flow->values[rhs - 1u];
  if (right.type != factor.type || right.count != rhs_count ||
      !same_fixed_storage_policy(right.fixed_format, factor.fixed_format)) {
    reject(*flow, Reason::SolveShapeMismatch);
    return {};
  }
  const std::uint32_t values =
      append(*flow, factor.type, rhs_count, factor.fixed_format);
  const std::uint32_t status = append(*flow, Type::U32, batches);
  if (values == 0u || status == 0u) {
    return {};
  }
  const std::array lu_inputs{packed, pivots, rhs};
  const std::array direct_inputs{packed, rhs};
  const std::span<const std::uint32_t> inputs =
      operation == FactorOp::Lu ? std::span<const std::uint32_t>{lu_inputs}
                                : std::span<const std::uint32_t>{direct_inputs};
  const std::array outputs{values, status};
  if (!append_primitive(*flow, inputs, outputs, Primitive::Solve,
                        {.first = rows,
                         .second = rhs_cols,
                         .third = batches,
                         .mode = static_cast<std::uint32_t>(operation),
                         .flag = true})) {
    return {};
  }
  flow->output = values;
  return SolveIds{values, status};
}

SolveIds flow_matrix_solve(const std::shared_ptr<FlowState> &flow,
                           const FactorOp operation, const std::uint32_t matrix,
                           const std::uint32_t rhs, const std::size_t rows,
                           const std::size_t cols, const std::size_t rhs_cols,
                           const std::size_t batches) {
  if (flow == nullptr || !flow->status || matrix == 0u || rhs == 0u ||
      matrix > flow->values.size() || rhs > flow->values.size()) {
    return {};
  }
  if (operation != FactorOp::Lu && operation != FactorOp::Qr &&
      operation != FactorOp::Cholesky) {
    reject(*flow, Reason::SolveFactorUnsupported);
    return {};
  }
  if (rows != cols) {
    reject(*flow, Reason::SolveShapeMismatch);
    return {};
  }
  std::size_t expected_matrix = 0u;
  std::size_t rhs_count = 0u;
  if (!matrix_count(*flow, rows, cols, batches, expected_matrix) ||
      !matrix_count(*flow, rows, rhs_cols, batches, rhs_count)) {
    return {};
  }
  const FlowValue &left = flow->values[matrix - 1u];
  const FlowValue &right = flow->values[rhs - 1u];
  if (!type_fixed(left.type) || left.count != expected_matrix ||
      right.type != left.type || right.count != rhs_count ||
      !same_fixed_storage_policy(right.fixed_format, left.fixed_format)) {
    reject(*flow, Reason::SolveShapeMismatch);
    return {};
  }
  FixedFormat output_format = left.fixed_format;
  output_format.approximation = Approximation::Deterministic;
  const std::uint32_t values =
      append(*flow, left.type, rhs_count, output_format);
  const std::uint32_t status = append(*flow, Type::U32, batches);
  if (values == 0u || status == 0u) {
    return {};
  }
  const std::array inputs{matrix, rhs};
  const std::array outputs{values, status};
  if (!append_primitive(*flow, inputs, outputs, Primitive::Solve,
                        {.first = rows,
                         .second = rhs_cols,
                         .third = batches,
                         .mode = static_cast<std::uint32_t>(operation),
                         .flag = false})) {
    return {};
  }
  flow->output = values;
  return SolveIds{values, status};
}

SpectrumIds flow_spectrum(const std::shared_ptr<FlowState> &flow,
                          const SpectrumOp operation,
                          const SpectrumVectors vector_mode,
                          const std::size_t rows, const std::size_t cols,
                          const std::size_t batches,
                          const std::uint32_t iterations) {
  if (flow == nullptr || !flow->status) {
    return {};
  }
  std::size_t input_count = 0u;
  if (!matrix_count(*flow, rows, cols, batches, input_count)) {
    return {};
  }
  if (input_count != flow_output_count(flow)) {
    reject(*flow, Reason::SpectrumShapeMismatch);
    return {};
  }
  if (operation == SpectrumOp::Eigen && rows != cols) {
    reject(*flow, Reason::SpectrumShapeSymmetric);
    return {};
  }
  const std::size_t width =
      operation == SpectrumOp::Svd ? std::min(rows, cols) : rows;
  std::size_t value_count = 0u;
  if (!matrix_count(*flow, width, 1u, batches, value_count)) {
    return {};
  }
  std::size_t vector_count = 0u;
  if (vector_mode == SpectrumVectors::Thin &&
      !matrix_count(*flow, rows, width, batches, vector_count)) {
    return {};
  }
  if (vector_mode == SpectrumVectors::Full &&
      !matrix_count(*flow, rows, rows, batches, vector_count)) {
    return {};
  }
  const FlowValue &source = flow->values[flow->output - 1u];
  const Type type = source.type;
  FixedFormat fixed_format = source.fixed_format;
  if (type_fixed(type)) {
    fixed_format.approximation = Approximation::Deterministic;
  }
  const std::uint32_t values = append(*flow, type, value_count, fixed_format);
  const std::uint32_t vectors =
      vector_count == 0u ? 0u : append(*flow, type, vector_count, fixed_format);
  const std::uint32_t status = append(*flow, Type::U32, batches);
  if (values == 0u || status == 0u || (vector_count != 0u && vectors == 0u)) {
    return {};
  }
  const std::array inputs{flow->output};
  const std::array full_outputs{values, vectors, status};
  const std::array values_outputs{values, status};
  const std::span<const std::uint32_t> outputs =
      vectors == 0u ? std::span<const std::uint32_t>{values_outputs}
                    : std::span<const std::uint32_t>{full_outputs};
  if (!append_primitive(*flow, inputs, outputs, Primitive::Spectrum,
                        {.first = rows,
                         .second = cols,
                         .third = batches,
                         .fourth = static_cast<std::uint64_t>(vector_mode),
                         .mode = static_cast<std::uint32_t>(operation),
                         .extra = iterations})) {
    return {};
  }
  flow->output = values;
  return SpectrumIds{values, vectors, status};
}

} // namespace rund::compute::detail
