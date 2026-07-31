#include "model.hpp"

#include "../../../fixed/format.hpp"
#include "../../../type.hpp"
#include "../../local.hpp"

#include <accel/graph/factory.hpp>

namespace rund::compute::detail::graph_build_detail {

using graph_detail::collective_block;

rund::AccelGraphNode make_node(const Primitive primitive,
                               const std::span<const GraphArg> inputs,
                               const PrimitiveOptions options,
                               std::uint32_t &primary_write) {
  const GraphArg *primary = inputs.empty() ? nullptr : &inputs.front();
  if (primitive == Primitive::Partition && inputs.size() == 2u) {
    primary = &inputs[1u];
  }
  const Type type = primary == nullptr ? Type::U32 : primary->type;
  const std::uint64_t count = primary == nullptr ? 0 : primary->count;
  const bool wide = type_bytes(type) == 8;
  const bool fixed = type == Type::FixedLane32 || type == Type::FixedLane64;
  const kernel::ComputeFixedFormat source_format =
      primary == nullptr ? kernel::ComputeFixedFormat{}
                         : kernel_format(primary->fixed_format);
  const kernel::ComputeFixedFormat exact_primitive_format =
      fixed ? kernel::PrimitiveFixedFormat(source_format,
                                           source_format.approximation)
            : kernel::ComputeFixedFormat{};
  const kernel::ComputeFixedFormat deterministic_primitive_format =
      fixed ? kernel::PrimitiveFixedFormat(
                  source_format, kernel::ComputeApproximation::Deterministic)
            : kernel::ComputeFixedFormat{};
  const kernel::MatrixArithmetic matrix_arithmetic =
      type == Type::FixedLane32 || type == Type::FixedLane64
          ? kernel::MatrixArithmetic::Fixed
          : (type == Type::I32 || type == Type::I64
                 ? kernel::MatrixArithmetic::SignedWrap
                 : kernel::MatrixArithmetic::UnsignedWrap);
  primary_write = primitive == Primitive::Argsort ? 1u : 0u;
  switch (primitive) {
  case Primitive::SegmentedScan: {
    const auto operation = segmented_scan_op(options.mode);
    if (!operation) {
      return {};
    }
    return rund::AccelSegmentedScan(
        nullptr, 0,
        kernel::SegmentedScanDesc{
            .op = *operation,
            .element = wide ? kernel::SegmentedScanElement::U64
                            : kernel::SegmentedScanElement::U32,
            .element_count = count,
            .block_size = collective_block(count),
        });
  }
  case Primitive::SegmentedReduce: {
    const auto operation = reduce_op(options.mode);
    if (!operation) {
      return {};
    }
    return rund::AccelSegmentedReduce(
        nullptr, 0,
        kernel::SegmentedReduceDesc{
            .op = *operation,
            .element =
                wide ? kernel::ReduceElement::U64 : kernel::ReduceElement::U32,
            .element_count = count,
            .block_size = collective_block(count),
        });
  }
  case Primitive::Sort:
  case Primitive::Argsort:
    return rund::AccelSort(
        nullptr, 0,
        kernel::SortDesc{
            .key = wide ? kernel::SortKey::U64 : kernel::SortKey::U32,
            .value = kernel::SortValue::IdentityU32,
            .element_count = count,
            .radix_bits = 8u,
            .key_bits = 0,
            .stable = true,
            .count_source = inputs.size() == 2u
                                ? (type_bytes(inputs[1u].type) == 8u
                                       ? kernel::ComputeCountSource::BufferU64
                                       : kernel::ComputeCountSource::BufferU32)
                                : kernel::ComputeCountSource::Descriptor,
        });
  case Primitive::Compact:
    return rund::AccelCompact(nullptr, 0,
                              kernel::CompactDesc{
                                  .element_count = count,
                                  .output_capacity = options.first,
                                  .flag_bytes = 4u,
                                  .output_bytes = 4u,
                              });
  case Primitive::Gather:
    return rund::AccelGather(
        nullptr, 0,
        kernel::GatherDesc{
            .element =
                wide ? kernel::GatherElement::U64 : kernel::GatherElement::U32,
            .element_count = inputs.size() > 1 ? inputs[1].count : 0,
            .source_count = count,
            .count_source = inputs.size() == 3u
                                ? (type_bytes(inputs[2u].type) == 8u
                                       ? kernel::ComputeCountSource::BufferU64
                                       : kernel::ComputeCountSource::BufferU32)
                                : kernel::ComputeCountSource::Descriptor,
        });
  case Primitive::Histogram:
    return rund::AccelHistogram(nullptr, 0,
                                kernel::HistogramDesc{
                                    .index = kernel::HistogramIndex::U32,
                                    .count = kernel::HistogramCount::U32,
                                    .element_count = count,
                                    .bin_count = options.first,
                                });
  case Primitive::Partition:
    return rund::AccelPartition(
        nullptr, 0,
        kernel::PartitionDesc{
            .element_count = count,
            .flag_bytes = static_cast<std::uint32_t>(
                inputs.empty() ? 4u : type_bytes(inputs[0u].type)),
            .value_bytes = static_cast<std::uint32_t>(type_bytes(type)),
        });
  case Primitive::Reduce: {
    const auto operation = reduce_op(options.mode, options.flag);
    if (!operation) {
      return {};
    }
    return rund::AccelReduce(
        nullptr, 0,
        kernel::ReduceDesc{
            .op = *operation,
            .element =
                wide ? kernel::ReduceElement::U64 : kernel::ReduceElement::U32,
            .element_count = count,
            .block_size = collective_block(count),
            .count_source = inputs.size() == 2u
                                ? (type_bytes(inputs[1u].type) == 8u
                                       ? kernel::ComputeCountSource::BufferU64
                                       : kernel::ComputeCountSource::BufferU32)
                                : kernel::ComputeCountSource::Descriptor,
        });
  }
  case Primitive::Scatter:
    return rund::AccelScatter(nullptr, 0,
                              kernel::ScatterDesc{
                                  .element = wide ? kernel::ScatterElement::U64
                                                  : kernel::ScatterElement::U32,
                                  .element_count = count,
                                  .output_count = options.first,
                              });
  case Primitive::ScatterReduce: {
    const auto operation = scatter_reduce_op(options.mode);
    if (!operation) {
      return {};
    }
    return rund::AccelScatterReduce(
        nullptr, 0,
        kernel::ScatterReduceDesc{
            .op = *operation,
            .domain = type_domain(type),
            .fixed_format = fixed ? deterministic_primitive_format
                                  : kernel::ComputeFixedFormat{},
            .element_count = count,
            .output_count = options.first,
            .count_source = inputs.size() == 3u
                                ? (type_bytes(inputs[2u].type) == 8u
                                       ? kernel::ComputeCountSource::BufferU64
                                       : kernel::ComputeCountSource::BufferU32)
                                : kernel::ComputeCountSource::Descriptor,
        });
  }
  case Primitive::Stencil: {
    const auto operation = stencil_op(options.mode);
    if (!operation) {
      return {};
    }
    return rund::AccelStencil(nullptr, 0,
                              kernel::StencilDesc{
                                  .op = *operation,
                                  .element = wide ? kernel::StencilElement::U64
                                                  : kernel::StencilElement::U32,
                                  .boundary = kernel::StencilBoundary::Clamp,
                                  .element_count = count,
                                  .radius = options.first,
                              });
  }
  case Primitive::Transform: {
    const auto direction = transform_direction(options.mode);
    if (!direction) {
      return {};
    }
    return rund::AccelTransform(
        nullptr, 0,
        kernel::TransformDesc{
            .op = kernel::TransformOp::Fourier,
            .direction = *direction,
            .layout = kernel::TransformLayout::Split,
            .normalization = options.flag ? kernel::TransformNorm::InverseLength
                                          : kernel::TransformNorm::None,
            .element_count = count,
            .fixed_format = deterministic_primitive_format,
        });
  }
  case Primitive::Matrix: {
    const auto operation = matrix_op(options.mode);
    if (!operation) {
      return {};
    }
    return rund::AccelMatrix(
        nullptr, 0,
        kernel::MatrixDesc{
            .op = *operation,
            .layout = kernel::MatrixLayout::RowMajor,
            .arithmetic = matrix_arithmetic,
            .rows = options.first,
            .cols = options.second,
            .inner = options.third,
            .batch_count = options.fourth,
            .element_bytes = static_cast<std::uint32_t>(type_bytes(type)),
            .fixed_format = exact_primitive_format,
        });
  }
  case Primitive::Factor: {
    const auto operation = factor_op(options.mode);
    if (!operation) {
      return {};
    }
    const kernel::FactorOp op = *operation;
    return rund::AccelFactor(
        nullptr, 0,
        kernel::FactorDesc{
            .op = op,
            .layout = kernel::MatrixLayout::RowMajor,
            .output = op == kernel::FactorOp::QR
                          ? kernel::FactorOutput::Separate
                          : kernel::FactorOutput::Packed,
            .pivot = op == kernel::FactorOp::Cholesky
                         ? kernel::PivotOp::None
                         : kernel::PivotOp::Partial,
            .rows = options.first,
            .cols = options.second,
            .batch_count = options.third,
            .element_bytes = static_cast<std::uint32_t>(type_bytes(type)),
            .fixed_format = deterministic_primitive_format,
        });
  }
  case Primitive::Solve: {
    const auto operation = factor_op(options.mode);
    if (!operation) {
      return {};
    }
    const kernel::FactorOp op = *operation;
    return rund::AccelSolve(
        nullptr, 0,
        kernel::SolveDesc{
            .op = kernel::SolveOp::Linear,
            .input = options.flag ? kernel::SolveInput::Factor
                                  : kernel::SolveInput::Matrix,
            .factor = op,
            .layout = kernel::MatrixLayout::RowMajor,
            .pivot = op == kernel::FactorOp::LU ? kernel::PivotOp::Partial
                                                : kernel::PivotOp::None,
            .rows = options.first,
            .rhs_cols = options.second,
            .batch_count = options.third,
            .element_bytes = static_cast<std::uint32_t>(type_bytes(type)),
            .fixed_format = deterministic_primitive_format,
        });
  }
  case Primitive::Spectrum: {
    const auto operation = spectrum_op(options.mode);
    const auto vectors = spectrum_vectors(options.fourth);
    if (!operation || !vectors) {
      return {};
    }
    const bool eigen = *operation == kernel::SpectrumOp::Eigen;
    return rund::AccelSpectrum(
        nullptr, 0,
        kernel::SpectrumDesc{
            .op = *operation,
            .domain = eigen ? kernel::SpectrumDomain::SymmetricReal
                            : kernel::SpectrumDomain::GeneralReal,
            .vectors = *vectors,
            .layout = kernel::MatrixLayout::RowMajor,
            .rows = options.first,
            .cols = options.second,
            .batch_count = options.third,
            .max_iterations = options.extra,
            .element_bytes = static_cast<std::uint32_t>(type_bytes(type)),
            .fixed_format = deterministic_primitive_format,
        });
  }
  }
  return {};
}

} // namespace rund::compute::detail::graph_build_detail
