#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>

#include <accel/graph/factory/buffer/read.hpp>
#include <accel/graph/factory/buffer/write.hpp>
#include <accel/graph/factory/primitive/transform/node.hpp>
#include <kernel/program/compute/transform/plan.hpp>
#include <kernel/program/compute/transform/reference.hpp>

#include "primitive/local.hpp"
#include "test/assert.hpp"
#include "test/compute/fixed.hpp"
#include <node/accel/context.hpp>
#include <node/accel/pick.hpp>

#include <array>
#include <iostream>

namespace node_accel_contract {
namespace {

[[nodiscard]] bool TransformFail(const char *const reason) {
  std::cerr << "transform backend match failed: " << reason << '\n';
  return false;
}

} // namespace

[[nodiscard]] bool BackendRunsTransform(const rund::AccelDevice &pick) {
  namespace fix = node_accel_contract::primitive;
  if (!pick.check.ok) {
    return TransformFail(pick.check.reason);
  }

  std::array<rund::kernel::i32, 2'048u> input_real{};
  std::array<rund::kernel::i32, 2'048u> input_imag{};
  for (std::size_t index = 0u; index < input_real.size(); ++index) {
    input_real[index] =
        (static_cast<rund::kernel::i32>((index * 5u + 3u) % 17u) - 8) *
        (1 << 18u);
    input_imag[index] =
        (static_cast<rund::kernel::i32>((index * 7u + 1u) % 13u) - 6) *
        (1 << 18u);
  }
  std::array<rund::kernel::i32, 2'048u> expected_real{};
  std::array<rund::kernel::i32, 2'048u> expected_imag{};
  const rund::kernel::TransformDesc desc{
      .op = rund::kernel::TransformOp::Fourier,
      .direction = rund::kernel::TransformDir::Forward,
      .layout = rund::kernel::TransformLayout::Split,
      .normalization = rund::kernel::TransformNorm::InverseLength,
      .element_count = input_real.size(),
      .fixed_format = rund::kernel::PrimitiveFixedFormat(
          test::FixedFormatForLane(rund::kernel::ComputeScalar::Lane32),
          rund::kernel::ComputeApproximation::Deterministic),
  };
  const rund::kernel::TransformPlan plan = rund::kernel::PlanTransform(desc);
  if (!rund::kernel::ReferenceFourierSplitI32(
           input_real.data(), input_imag.data(), expected_real.data(),
           expected_imag.data(), plan)
           .ok) {
    return TransformFail("reference");
  }

  auto context = rund::node::accel::OpenAccel(pick);
  if (!context.check.ok) {
    return TransformFail(context.check.reason);
  }
  auto real_in = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::ReadOnly,
                               sizeof(rund::kernel::i32), input_real.size()));
  auto imag_in = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::ReadOnly,
                               sizeof(rund::kernel::i32), input_imag.size()));
  auto real_out = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::WriteOnly,
                               sizeof(rund::kernel::i32), input_real.size()));
  auto imag_out = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::WriteOnly,
                               sizeof(rund::kernel::i32), input_imag.size()));
  if (!real_in.check.ok || !imag_in.check.ok || !real_out.check.ok ||
      !imag_out.check.ok ||
      !rund::node::accel::UploadAccelBuffer(context, real_in, input_real.data(),
                                            input_real.size() *
                                                sizeof(rund::kernel::i32))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(context, imag_in, input_imag.data(),
                                            input_imag.size() *
                                                sizeof(rund::kernel::i32))
           .ok) {
    return TransformFail("buffer");
  }

  const std::array<rund::AccelGraphBufferRef, 4u> refs{
      rund::AccelGraphBufferRef{.buffer = &real_in,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &imag_in,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &real_out,
                                .role = rund::kernel::BufferRole::Write},
      rund::AccelGraphBufferRef{.buffer = &imag_out,
                                .role = rund::kernel::BufferRole::Write},
  };
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelTransform(refs.data(), refs.size(), desc),
  };
  if (nodes[0].kind != rund::kernel::NodeKind::Transform) {
    return TransformFail("factory.kind");
  }
  if (nodes[0].element_count != input_real.size()) {
    return TransformFail("factory.count");
  }
  const rund::kernel::TransformHash hash =
      rund::kernel::HashTransform(nodes[0].transform);
  if (nodes[0].primitive_hash_hi != hash.hi ||
      nodes[0].primitive_hash_lo != hash.lo) {
    return TransformFail("factory.hash");
  }
  if (nodes[0].ir != nullptr || nodes[0].sort.key_bits != 0u) {
    return TransformFail("factory.payload");
  }
  if (nodes[0].scan.element_count != 0u ||
      nodes[0].scan.op != rund::kernel::ScanOp::ExclusiveSum ||
      nodes[0].scan.element != rund::kernel::ScanElement::U32 ||
      nodes[0].scan.block_size != 0u ||
      nodes[0].segmented_scan.element_count != 0u ||
      nodes[0].segmented_scan.op !=
          rund::kernel::SegmentedScanOp::ExclusiveSum ||
      nodes[0].segmented_scan.element !=
          rund::kernel::SegmentedScanElement::U32 ||
      nodes[0].segmented_scan.block_size != 0u ||
      nodes[0].segmented_reduce.element_count != 0u ||
      nodes[0].segmented_reduce.op != rund::kernel::ReduceOp::Sum ||
      nodes[0].segmented_reduce.element != rund::kernel::ReduceElement::U32 ||
      nodes[0].segmented_reduce.block_size != 0u ||
      nodes[0].gather.element_count != 0u ||
      nodes[0].gather.element != rund::kernel::GatherElement::U32 ||
      nodes[0].gather.source_count != 0u ||
      nodes[0].histogram.element_count != 0u ||
      nodes[0].histogram.index != rund::kernel::HistogramIndex::U32 ||
      nodes[0].histogram.count != rund::kernel::HistogramCount::U32 ||
      nodes[0].histogram.bin_count != 0u ||
      nodes[0].partition.element_count != 0u ||
      nodes[0].partition.flag_bytes != sizeof(rund::kernel::u32) ||
      nodes[0].partition.value_bytes != sizeof(rund::kernel::u32) ||
      nodes[0].reduce.element_count != 0u ||
      nodes[0].reduce.op != rund::kernel::ReduceOp::Sum ||
      nodes[0].reduce.element != rund::kernel::ReduceElement::U32 ||
      nodes[0].reduce.block_size != 0u ||
      nodes[0].scatter.element_count != 0u ||
      nodes[0].scatter.element != rund::kernel::ScatterElement::U32 ||
      nodes[0].scatter.output_count != 0u ||
      nodes[0].stencil.element_count != 0u ||
      nodes[0].stencil.op != rund::kernel::StencilOp::Sum ||
      nodes[0].stencil.element != rund::kernel::StencilElement::U32 ||
      nodes[0].stencil.boundary != rund::kernel::StencilBoundary::Clamp ||
      nodes[0].stencil.radius != 1u ||
      nodes[0].matrix.op != rund::kernel::MatrixOp::Mul ||
      nodes[0].matrix.layout != rund::kernel::MatrixLayout::RowMajor ||
      nodes[0].matrix.rows != 0u || nodes[0].matrix.cols != 0u ||
      nodes[0].matrix.inner != 0u || nodes[0].matrix.batch_count != 1u ||
      nodes[0].matrix.element_bytes != sizeof(rund::kernel::u32)) {
    return TransformFail("factory.default_payload");
  }
  if (nodes[0].buffer_count != 4u || nodes[0].buffers == nullptr ||
      refs[0].role != rund::kernel::BufferRole::Read ||
      refs[1].role != rund::kernel::BufferRole::Read ||
      refs[2].role != rund::kernel::BufferRole::Write ||
      refs[3].role != rund::kernel::BufferRole::Write ||
      refs[0].buffer->scalar_width_bytes != sizeof(rund::kernel::i32) ||
      refs[1].buffer->scalar_width_bytes != sizeof(rund::kernel::i32) ||
      refs[2].buffer->scalar_width_bytes != sizeof(rund::kernel::i32) ||
      refs[3].buffer->scalar_width_bytes != sizeof(rund::kernel::i32) ||
      refs[0].buffer->count != input_real.size() ||
      refs[1].buffer->count != input_real.size() ||
      refs[2].buffer->count != input_real.size() ||
      refs[3].buffer->count != input_real.size() ||
      refs[0].buffer->usage != rund::BufferUsage::ReadOnly ||
      refs[1].buffer->usage != rund::BufferUsage::ReadOnly ||
      refs[2].buffer->usage != rund::BufferUsage::WriteOnly ||
      refs[3].buffer->usage != rund::BufferUsage::WriteOnly) {
    return TransformFail("factory.binding_shape");
  }
  auto kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = nodes.data(),
                   .node_count = nodes.size(),
                   .scalar = rund::kernel::ComputeScalar::Lane32,
                   .domain = rund::kernel::ComputeDomain::Fixed,
                   .fixed_format = desc.fixed_format,
               });
  if (!kernel.check.ok) {
    return TransformFail(kernel.check.reason);
  }
  const std::array<rund::AccelRunBinding, 4u> bindings{
      rund::AccelRunBinding{.buffer = &real_in,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &imag_in,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &real_out,
                            .role = rund::kernel::BufferRole::Write},
      rund::AccelRunBinding{.buffer = &imag_out,
                            .role = rund::kernel::BufferRole::Write},
  };
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(context, kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = input_real.size(),
                                            .fresh_evidence = true,
                                        });
  if (!evidence.ok || evidence.dispatch_count != plan.pass_count ||
      evidence.original_dispatch_count != plan.pass_count ||
      evidence.final_dispatch_count != plan.pass_count) {
    return TransformFail(evidence.reason);
  }

  std::array<rund::kernel::i32, 2'048u> real_download{};
  std::array<rund::kernel::i32, 2'048u> imag_download{};
  if (!rund::node::accel::DownloadAccelBuffer(
           context, real_out, real_download.data(),
           real_download.size() * sizeof(rund::kernel::i32))
           .ok ||
      !rund::node::accel::DownloadAccelBuffer(
           context, imag_out, imag_download.data(),
           imag_download.size() * sizeof(rund::kernel::i32))
           .ok) {
    return TransformFail("download");
  }
  if (fix::HashValues(real_download.data(), real_download.size()) !=
          fix::HashValues(expected_real.data(), expected_real.size()) ||
      fix::HashValues(imag_download.data(), imag_download.size()) !=
          fix::HashValues(expected_imag.data(), expected_imag.size())) {
    return TransformFail("mismatch");
  }
  return true;
}

[[nodiscard]] bool BackendRunsTransform64(const rund::AccelDevice &pick) {
  namespace fix = node_accel_contract::primitive;
  std::array<rund::kernel::i64, 2'048u> input_real{};
  std::array<rund::kernel::i64, 2'048u> input_imag{};
  for (std::size_t index = 0u; index < input_real.size(); ++index) {
    input_real[index] =
        (static_cast<rund::kernel::i64>((index * 5u + 3u) % 17u) - 8) *
        (rund::kernel::i64{1} << 48u);
    input_imag[index] =
        (static_cast<rund::kernel::i64>((index * 7u + 1u) % 13u) - 6) *
        (rund::kernel::i64{1} << 48u);
  }
  std::array<rund::kernel::i64, 2'048u> expected_real{};
  std::array<rund::kernel::i64, 2'048u> expected_imag{};
  const rund::kernel::TransformDesc desc{
      .op = rund::kernel::TransformOp::Fourier,
      .direction = rund::kernel::TransformDir::Forward,
      .layout = rund::kernel::TransformLayout::Split,
      .normalization = rund::kernel::TransformNorm::Unitary,
      .element_count = input_real.size(),
      .fixed_format = rund::kernel::PrimitiveFixedFormat(
          test::FixedFormatForLane(rund::kernel::ComputeScalar::Lane64),
          rund::kernel::ComputeApproximation::Deterministic),
  };
  const rund::kernel::TransformPlan plan = rund::kernel::PlanTransform(desc);
  if (!rund::kernel::ReferenceFourierSplitI64(
           input_real.data(), input_imag.data(), expected_real.data(),
           expected_imag.data(), plan)
           .ok) {
    return TransformFail("reference64");
  }
  auto context = rund::node::accel::OpenAccel(pick);
  if (!context.check.ok) {
    return TransformFail(context.check.reason);
  }
  auto real_in = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::ReadOnly,
                               sizeof(rund::kernel::i64), input_real.size()));
  auto imag_in = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::ReadOnly,
                               sizeof(rund::kernel::i64), input_imag.size()));
  auto real_out = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::WriteOnly,
                               sizeof(rund::kernel::i64), input_real.size()));
  auto imag_out = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::WriteOnly,
                               sizeof(rund::kernel::i64), input_imag.size()));
  if (!real_in.check.ok || !imag_in.check.ok || !real_out.check.ok ||
      !imag_out.check.ok ||
      !rund::node::accel::UploadAccelBuffer(context, real_in, input_real.data(),
                                            input_real.size() *
                                                sizeof(rund::kernel::i64))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(context, imag_in, input_imag.data(),
                                            input_imag.size() *
                                                sizeof(rund::kernel::i64))
           .ok) {
    return TransformFail("buffer64");
  }
  const std::array<rund::AccelGraphBufferRef, 4u> refs{
      rund::AccelRead(real_in, "real"),
      rund::AccelRead(imag_in, "imag"),
      rund::AccelWrite(real_out, "real_out"),
      rund::AccelWrite(imag_out, "imag_out"),
  };
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelTransform(refs.data(), refs.size(), desc),
  };
  auto kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = nodes.data(),
                   .node_count = nodes.size(),
                   .scalar = rund::kernel::ComputeScalar::Lane64,
                   .domain = rund::kernel::ComputeDomain::Fixed,
                   .fixed_format = desc.fixed_format,
               });
  if (!kernel.check.ok) {
    return TransformFail(kernel.check.reason);
  }
  const std::array<rund::AccelRunBinding, 4u> bindings{
      rund::AccelRunBinding{.buffer = &real_in,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &imag_in,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &real_out,
                            .role = rund::kernel::BufferRole::Write},
      rund::AccelRunBinding{.buffer = &imag_out,
                            .role = rund::kernel::BufferRole::Write},
  };
  const auto evidence =
      rund::node::accel::RunAccelKernel(context, kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = input_real.size(),
                                            .fresh_evidence = true,
                                        });
  std::array<rund::kernel::i64, 2'048u> real_download{};
  std::array<rund::kernel::i64, 2'048u> imag_download{};
  return evidence.ok && evidence.dispatch_count == plan.pass_count &&
         evidence.original_dispatch_count == plan.pass_count &&
         evidence.final_dispatch_count == plan.pass_count &&
         rund::node::accel::DownloadAccelBuffer(
             context, real_out, real_download.data(),
             real_download.size() * sizeof(rund::kernel::i64))
             .ok &&
         rund::node::accel::DownloadAccelBuffer(
             context, imag_out, imag_download.data(),
             imag_download.size() * sizeof(rund::kernel::i64))
             .ok &&
         fix::HashValues(real_download.data(), real_download.size()) ==
             fix::HashValues(expected_real.data(), expected_real.size()) &&
         fix::HashValues(imag_download.data(), imag_download.size()) ==
             fix::HashValues(expected_imag.data(), expected_imag.size());
}

[[nodiscard]] bool
TransformRejectsNonPowerOfTwo(const rund::AccelDevice &pick) {
  namespace fix = node_accel_contract::primitive;
  auto context = rund::node::accel::OpenAccel(pick);
  if (!context.check.ok) {
    return false;
  }
  auto in = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::ReadOnly,
                               sizeof(rund::kernel::i32), 3u));
  auto out = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::WriteOnly,
                               sizeof(rund::kernel::i32), 3u));
  const std::array<rund::AccelGraphBufferRef, 4u> refs{
      rund::AccelGraphBufferRef{.buffer = &in,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &in,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &out,
                                .role = rund::kernel::BufferRole::Write},
      rund::AccelGraphBufferRef{.buffer = &out,
                                .role = rund::kernel::BufferRole::Write},
  };
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelTransform(
          refs.data(), refs.size(),
          rund::kernel::TransformDesc{
              .op = rund::kernel::TransformOp::Fourier,
              .direction = rund::kernel::TransformDir::Forward,
              .layout = rund::kernel::TransformLayout::Split,
              .normalization = rund::kernel::TransformNorm::None,
              .element_count = 3u,
          }),
  };
  const auto kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = nodes.data(),
                   .node_count = nodes.size(),
                   .scalar = rund::kernel::ComputeScalar::Lane32,
                   .domain = rund::kernel::ComputeDomain::Fixed,
                   .fixed_format = test::FixedFormatForLane(
                       rund::kernel::ComputeScalar::Lane32),
               });
  return !kernel.check.ok;
}

[[nodiscard]] bool AvailableBackendsRunTransformNatively() {
  namespace fix = node_accel_contract::primitive;
  for (const rund::AccelApi api :
       {rund::AccelApi::Metal, rund::AccelApi::Vulkan}) {
    const rund::AccelDevice pick =
        rund::node::accel::PickAccel(fix::Policy(api));
    if (!pick.check.ok) {
      continue;
    }
    if (pick.api == rund::AccelApi::Cpu) {
      return false;
    }
    if (!BackendRunsTransform(pick) || !BackendRunsTransform64(pick)) {
      return false;
    }
  }
  return true;
}

} // namespace node_accel_contract
