#include "defaults.hpp"

namespace rund::node::accel::detail {

bool DefaultScanDescriptor(const rund::kernel::ScanDesc& desc) noexcept {
  return desc.op == rund::kernel::ScanOp::ExclusiveSum &&
         desc.element == rund::kernel::ScanElement::U32 &&
         desc.element_count == 0u && desc.block_size == 0u;
}

bool DefaultSegmentedScanDescriptor(const rund::kernel::SegmentedScanDesc& desc) noexcept {
  return desc.op == rund::kernel::SegmentedScanOp::ExclusiveSum &&
         desc.element == rund::kernel::SegmentedScanElement::U32 &&
         desc.element_count == 0u && desc.block_size == 0u;
}

bool DefaultSegmentedReduceDescriptor(const rund::kernel::SegmentedReduceDesc& desc) noexcept {
  return desc.op == rund::kernel::ReduceOp::Sum &&
         desc.element == rund::kernel::ReduceElement::U32 &&
         desc.element_count == 0u && desc.block_size == 0u;
}

bool DefaultGatherDescriptor(const rund::kernel::GatherDesc& desc) noexcept {
  return desc.element == rund::kernel::GatherElement::U32 &&
         desc.element_count == 0u && desc.source_count == 0u &&
         desc.count_source == rund::kernel::ComputeCountSource::Descriptor;
}

bool DefaultHistogramDescriptor(const rund::kernel::HistogramDesc& desc) noexcept {
  return desc.index == rund::kernel::HistogramIndex::U32 &&
         desc.count == rund::kernel::HistogramCount::U32 &&
         desc.element_count == 0u && desc.bin_count == 0u;
}

bool DefaultPartitionDescriptor(
    const rund::kernel::PartitionDesc& desc) noexcept {
  return desc.element_count == 0u &&
         desc.flag_bytes == sizeof(rund::kernel::u32) &&
         desc.value_bytes == sizeof(rund::kernel::u32);
}

bool DefaultReduceDescriptor(const rund::kernel::ReduceDesc& desc) noexcept {
  return desc.op == rund::kernel::ReduceOp::Sum &&
         desc.element == rund::kernel::ReduceElement::U32 &&
         desc.element_count == 0u && desc.block_size == 0u;
}

bool DefaultScatterDescriptor(
    const rund::kernel::ScatterDesc& desc) noexcept {
  return desc.element == rund::kernel::ScatterElement::U32 &&
         desc.element_count == 0u && desc.output_count == 0u;
}

bool DefaultScatterReduceDescriptor(
    const rund::kernel::ScatterReduceDesc &desc) noexcept {
  return desc.op == rund::kernel::ScatterReduceOp::Sum &&
         desc.domain == rund::kernel::ComputeDomain::U32 &&
         rund::kernel::ComputeFixedFormatAbsent(desc.fixed_format) &&
         desc.element_count == 0u && desc.output_count == 0u &&
         desc.count_source == rund::kernel::ComputeCountSource::Descriptor;
}

bool DefaultStencilDescriptor(
    const rund::kernel::StencilDesc& desc) noexcept {
  return desc.op == rund::kernel::StencilOp::Sum &&
         desc.element == rund::kernel::StencilElement::U32 &&
         desc.boundary == rund::kernel::StencilBoundary::Clamp &&
         desc.element_count == 0u && desc.radius == 1u;
}

bool DefaultTransformDescriptor(
    const rund::kernel::TransformDesc& desc) noexcept {
  return desc.op == rund::kernel::TransformOp::Fourier &&
         desc.direction == rund::kernel::TransformDir::Forward &&
         desc.layout == rund::kernel::TransformLayout::Split &&
         desc.normalization == rund::kernel::TransformNorm::None &&
         desc.element_count == 0u;
}

bool DefaultMatrixDescriptor(
    const rund::kernel::MatrixDesc& desc) noexcept {
  return desc.op == rund::kernel::MatrixOp::Mul &&
         desc.layout == rund::kernel::MatrixLayout::RowMajor &&
         desc.rows == 0u && desc.cols == 0u && desc.inner == 0u &&
         desc.batch_count == 1u && desc.element_bytes == 4u;
}

bool DefaultFactorDescriptor(
    const rund::kernel::FactorDesc& desc) noexcept {
  return desc.op == rund::kernel::FactorOp::LU &&
         desc.layout == rund::kernel::MatrixLayout::RowMajor &&
         desc.output == rund::kernel::FactorOutput::Packed &&
         desc.pivot == rund::kernel::PivotOp::Partial &&
         desc.rows == 0u && desc.cols == 0u && desc.batch_count == 1u &&
         desc.element_bytes == 4u;
}

bool DefaultSolveDescriptor(
    const rund::kernel::SolveDesc& desc) noexcept {
  return desc.op == rund::kernel::SolveOp::Linear &&
         desc.input == rund::kernel::SolveInput::Matrix &&
         desc.factor == rund::kernel::FactorOp::LU &&
         desc.layout == rund::kernel::MatrixLayout::RowMajor &&
         desc.pivot == rund::kernel::PivotOp::Partial &&
         desc.rows == 0u && desc.rhs_cols == 0u &&
         desc.batch_count == 1u && desc.element_bytes == 4u;
}

bool DefaultSpectrumDescriptor(
    const rund::kernel::SpectrumDesc& desc) noexcept {
  return desc.op == rund::kernel::SpectrumOp::SVD &&
         desc.domain == rund::kernel::SpectrumDomain::GeneralReal &&
         desc.vectors == rund::kernel::SpectrumVectors::ValuesOnly &&
         desc.layout == rund::kernel::MatrixLayout::RowMajor &&
         desc.rows == 0u && desc.cols == 0u && desc.batch_count == 1u &&
         desc.max_iterations == 32u && desc.element_bytes == 4u;
}

}  // namespace rund::node::accel::detail
