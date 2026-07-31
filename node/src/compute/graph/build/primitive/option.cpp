#include "model.hpp"

namespace rund::compute::detail::graph_build_detail {

std::optional<kernel::ReduceOp> reduce_op(const std::uint32_t mode,
                                          const bool count) noexcept {
  if (count) {
    return kernel::ReduceOp::CountNonzero;
  }
  switch (static_cast<Reduce>(mode)) {
  case Reduce::Sum:
    return kernel::ReduceOp::Sum;
  case Reduce::Min:
    return kernel::ReduceOp::Min;
  case Reduce::Max:
    return kernel::ReduceOp::Max;
  }
  return std::nullopt;
}

std::optional<kernel::ScatterReduceOp>
scatter_reduce_op(const std::uint32_t mode) noexcept {
  switch (static_cast<Reduce>(mode)) {
  case Reduce::Sum:
    return kernel::ScatterReduceOp::Sum;
  case Reduce::Min:
    return kernel::ScatterReduceOp::Min;
  case Reduce::Max:
    return kernel::ScatterReduceOp::Max;
  }
  return std::nullopt;
}

std::optional<kernel::StencilOp> stencil_op(const std::uint32_t mode) noexcept {
  switch (static_cast<Window>(mode)) {
  case Window::Sum:
    return kernel::StencilOp::Sum;
  case Window::Min:
    return kernel::StencilOp::Min;
  case Window::Max:
    return kernel::StencilOp::Max;
  }
  return std::nullopt;
}

std::optional<kernel::FactorOp> factor_op(const std::uint32_t mode) noexcept {
  switch (static_cast<FactorOp>(mode)) {
  case FactorOp::Lu:
    return kernel::FactorOp::LU;
  case FactorOp::Qr:
    return kernel::FactorOp::QR;
  case FactorOp::Cholesky:
    return kernel::FactorOp::Cholesky;
  }
  return std::nullopt;
}

std::optional<kernel::SegmentedScanOp>
segmented_scan_op(const std::uint32_t mode) noexcept {
  switch (static_cast<Scan>(mode)) {
  case Scan::InclusiveSum:
    return kernel::SegmentedScanOp::InclusiveSum;
  case Scan::ExclusiveSum:
    return kernel::SegmentedScanOp::ExclusiveSum;
  }
  return std::nullopt;
}

std::optional<kernel::TransformDir>
transform_direction(const std::uint32_t mode) noexcept {
  switch (static_cast<Direction>(mode)) {
  case Direction::Forward:
    return kernel::TransformDir::Forward;
  case Direction::Inverse:
    return kernel::TransformDir::Inverse;
  }
  return std::nullopt;
}

std::optional<kernel::MatrixOp> matrix_op(const std::uint32_t mode) noexcept {
  switch (static_cast<detail::MatrixMode>(mode)) {
  case detail::MatrixMode::Mul:
    return kernel::MatrixOp::Mul;
  case detail::MatrixMode::Transpose:
    return kernel::MatrixOp::Transpose;
  case detail::MatrixMode::BatchMul:
    return kernel::MatrixOp::BatchMul;
  }
  return std::nullopt;
}

std::optional<kernel::SpectrumOp>
spectrum_op(const std::uint32_t mode) noexcept {
  switch (static_cast<SpectrumOp>(mode)) {
  case SpectrumOp::Svd:
    return kernel::SpectrumOp::SVD;
  case SpectrumOp::Eigen:
    return kernel::SpectrumOp::Eigen;
  }
  return std::nullopt;
}

std::optional<kernel::SpectrumVectors>
spectrum_vectors(const std::uint64_t mode) noexcept {
  switch (static_cast<SpectrumVectors>(mode)) {
  case SpectrumVectors::Values:
    return kernel::SpectrumVectors::ValuesOnly;
  case SpectrumVectors::Thin:
    return kernel::SpectrumVectors::Thin;
  case SpectrumVectors::Full:
    return kernel::SpectrumVectors::Full;
  }
  return std::nullopt;
}

const char *unsupported(const Primitive primitive,
                        const PrimitiveOptions options) noexcept {
  switch (primitive) {
  case Primitive::SegmentedScan:
    return segmented_scan_op(options.mode) ? nullptr
                                           : "compute_scan_op_unsupported";
  case Primitive::SegmentedReduce:
  case Primitive::Reduce:
    return reduce_op(options.mode,
                     primitive == Primitive::Reduce && options.flag)
               ? nullptr
               : "compute_reduce_op_unsupported";
  case Primitive::ScatterReduce:
    return scatter_reduce_op(options.mode)
               ? nullptr
               : "compute_scatter_reduce_op_unsupported";
  case Primitive::Stencil:
    return stencil_op(options.mode) ? nullptr
                                    : "compute_stencil_op_unsupported";
  case Primitive::Transform:
    return transform_direction(options.mode)
               ? nullptr
               : "compute_transform_direction_unsupported";
  case Primitive::Matrix:
    return matrix_op(options.mode) ? nullptr : "compute_matrix_op_unsupported";
  case Primitive::Factor:
  case Primitive::Solve:
    return factor_op(options.mode) ? nullptr : "compute_factor_op_unsupported";
  case Primitive::Spectrum:
    if (!spectrum_op(options.mode)) {
      return "compute_spectrum_op_unsupported";
    }
    return spectrum_vectors(options.fourth)
               ? nullptr
               : "compute_spectrum_vectors_unsupported";
  case Primitive::Sort:
  case Primitive::Argsort:
  case Primitive::Compact:
  case Primitive::Gather:
  case Primitive::Histogram:
  case Primitive::Partition:
  case Primitive::Scatter:
    return nullptr;
  }
  return "compute_primitive_unsupported";
}

} // namespace rund::compute::detail::graph_build_detail
