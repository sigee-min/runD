#include "model.hpp"

namespace node_accel_contract {
namespace {

using namespace solve_contract;

template <typename Value>
[[nodiscard]] bool RunFactorReuseSolve(const rund::AccelDevice &pick,
                                       const rund::kernel::FactorOp factor) {
  namespace fix = node_accel_contract::primitive;
  const rund::kernel::ComputeScalar scalar =
      sizeof(Value) == sizeof(rund::kernel::i64)
          ? rund::kernel::ComputeScalar::Lane64
          : rund::kernel::ComputeScalar::Lane32;
  auto context = rund::node::accel::OpenAccel(pick);
  SolveBuffers b{};
  const std::uint64_t factor_count =
      factor == rund::kernel::FactorOp::QR ? 8u : 4u;
  if (!context.check.ok ||
      !CreateSolveBuffers<Value>(context, b, factor_count)) {
    return SolveReuseFail<Value>("buffer", factor);
  }
  const std::array<Value, 4u> matrix{
      static_cast<Value>(sizeof(Value) == sizeof(rund::kernel::i64) ? kWideOne
                                                                    : kOne),
      0, 0,
      static_cast<Value>(sizeof(Value) == sizeof(rund::kernel::i64) ? kWideOne
                                                                    : kOne)};
  const std::array<Value, 2u> rhs{
      static_cast<Value>(sizeof(Value) == sizeof(rund::kernel::i64) ? kWideHalf
                                                                    : kHalf),
      static_cast<Value>(sizeof(Value) == sizeof(rund::kernel::i64)
                             ? kWideQuarter
                             : kQuarter)};
  if (!rund::node::accel::UploadAccelBuffer(context, b.matrix, matrix.data(),
                                            matrix.size() * sizeof(Value))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(context, b.rhs, rhs.data(),
                                            rhs.size() * sizeof(Value))
           .ok) {
    return SolveReuseFail<Value>("upload", factor);
  }
  if (factor == rund::kernel::FactorOp::LU) {
    const std::array<rund::AccelGraphBufferRef, 4u> refs{
        rund::AccelRead(b.matrix, "matrix"),
        rund::AccelWrite(b.factor, "factor"), rund::AccelWrite(b.aux, "aux"),
        rund::AccelWrite(b.status, "status")};
    const std::array<rund::AccelGraphNode, 1u> nodes{
        rund::AccelFactor(refs.data(), refs.size(),
                          rund::kernel::FactorDesc{
                              .op = factor,
                              .layout = rund::kernel::MatrixLayout::RowMajor,
                              .output = rund::kernel::FactorOutput::Packed,
                              .pivot = rund::kernel::PivotOp::Partial,
                              .rows = 2u,
                              .cols = 2u,
                              .element_bytes = sizeof(Value),
                              .fixed_format = FixedFormat(scalar),
                          }),
    };
    const auto kernel = rund::node::accel::CompileAccelKernel(
        context, rund::AccelGraph{
                     .nodes = nodes.data(),
                     .node_count = nodes.size(),
                     .scalar = scalar,
                     .domain = rund::kernel::ComputeDomain::Fixed,
                     .fixed_format = test::FixedFormatForLane(scalar),
                 });
    const std::array<rund::AccelRunBinding, 4u> binds{
        rund::AccelRunBinding{.buffer = &b.matrix,
                              .role = rund::kernel::BufferRole::Read},
        rund::AccelRunBinding{.buffer = &b.factor,
                              .role = rund::kernel::BufferRole::Write},
        rund::AccelRunBinding{.buffer = &b.aux,
                              .role = rund::kernel::BufferRole::Write},
        rund::AccelRunBinding{.buffer = &b.status,
                              .role = rund::kernel::BufferRole::Write},
    };
    if (!kernel.check.ok ||
        !rund::node::accel::RunAccelKernel(context, kernel,
                                           rund::AccelRun{
                                               .bindings = binds.data(),
                                               .binding_count = binds.size(),
                                               .tile_count = 4u,
                                               .fresh_evidence = true,
                                           })
             .ok) {
      return SolveReuseFail<Value>(kernel.check.reason, factor);
    }
  } else {
    const std::array<rund::AccelGraphBufferRef, 3u> refs{
        rund::AccelRead(b.matrix, "matrix"),
        rund::AccelWrite(b.factor, "factor"),
        rund::AccelWrite(b.status, "status")};
    const std::array<rund::AccelGraphNode, 1u> nodes{
        rund::AccelFactor(
            refs.data(), refs.size(),
            rund::kernel::FactorDesc{
                .op = factor,
                .output = factor == rund::kernel::FactorOp::QR
                              ? rund::kernel::FactorOutput::Separate
                              : rund::kernel::FactorOutput::Packed,
                .pivot = rund::kernel::PivotOp::None,
                .rows = 2u,
                .cols = 2u,
                .element_bytes = sizeof(Value),
                .fixed_format = FixedFormat(scalar)}),
    };
    const auto kernel = rund::node::accel::CompileAccelKernel(
        context, rund::AccelGraph{
                     .nodes = nodes.data(),
                     .node_count = nodes.size(),
                     .scalar = scalar,
                     .domain = rund::kernel::ComputeDomain::Fixed,
                     .fixed_format = test::FixedFormatForLane(scalar),
                 });
    const std::array<rund::AccelRunBinding, 3u> binds{
        rund::AccelRunBinding{.buffer = &b.matrix,
                              .role = rund::kernel::BufferRole::Read},
        rund::AccelRunBinding{.buffer = &b.factor,
                              .role = rund::kernel::BufferRole::Write},
        rund::AccelRunBinding{.buffer = &b.status,
                              .role = rund::kernel::BufferRole::Write},
    };
    if (!kernel.check.ok ||
        !rund::node::accel::RunAccelKernel(context, kernel,
                                           rund::AccelRun{
                                               .bindings = binds.data(),
                                               .binding_count = binds.size(),
                                               .tile_count = 4u,
                                               .fresh_evidence = true,
                                           })
             .ok) {
      return SolveReuseFail<Value>(kernel.check.reason, factor);
    }
  }

  const auto solve_desc =
      rund::kernel::SolveDesc{.op = rund::kernel::SolveOp::Linear,
                              .input = rund::kernel::SolveInput::Factor,
                              .factor = factor,
                              .pivot = factor == rund::kernel::FactorOp::LU
                                           ? rund::kernel::PivotOp::Partial
                                           : rund::kernel::PivotOp::None,
                              .rows = 2u,
                              .rhs_cols = 1u,
                              .element_bytes = sizeof(Value),
                              .fixed_format = FixedFormat(scalar)};
  std::array<rund::kernel::u32, 1u> status{};
  if (factor == rund::kernel::FactorOp::LU) {
    const std::array<rund::AccelGraphBufferRef, 5u> refs{
        rund::AccelRead(b.factor, "factor"), rund::AccelRead(b.aux, "aux"),
        rund::AccelRead(b.rhs, "rhs"), rund::AccelWrite(b.output, "output"),
        rund::AccelWrite(b.status, "status")};
    const std::array<rund::AccelGraphNode, 1u> nodes{
        rund::AccelSolve(refs.data(), refs.size(), solve_desc),
    };
    const auto kernel = rund::node::accel::CompileAccelKernel(
        context, rund::AccelGraph{
                     .nodes = nodes.data(),
                     .node_count = nodes.size(),
                     .scalar = scalar,
                     .domain = rund::kernel::ComputeDomain::Fixed,
                     .fixed_format = test::FixedFormatForLane(scalar),
                 });
    const std::array<rund::AccelRunBinding, 5u> binds{
        rund::AccelRunBinding{.buffer = &b.factor,
                              .role = rund::kernel::BufferRole::Read},
        rund::AccelRunBinding{.buffer = &b.aux,
                              .role = rund::kernel::BufferRole::Read},
        rund::AccelRunBinding{.buffer = &b.rhs,
                              .role = rund::kernel::BufferRole::Read},
        rund::AccelRunBinding{.buffer = &b.output,
                              .role = rund::kernel::BufferRole::Write},
        rund::AccelRunBinding{.buffer = &b.status,
                              .role = rund::kernel::BufferRole::Write},
    };
    const auto evidence =
        rund::node::accel::RunAccelKernel(context, kernel,
                                          rund::AccelRun{
                                              .bindings = binds.data(),
                                              .binding_count = binds.size(),
                                              .tile_count = 2u,
                                              .fresh_evidence = true,
                                          });
    constexpr std::uint64_t expected_dispatches = 1u;
    if (!evidence.ok || evidence.dispatch_count != expected_dispatches ||
        evidence.failed_batches != 0u ||
        !rund::node::accel::DownloadAccelBuffer(
             context, b.status, status.data(), sizeof(rund::kernel::u32))
             .ok ||
        status[0] !=
            static_cast<rund::kernel::u32>(rund::kernel::SolveStatus::Ok)) {
      return SolveReuseFail<Value>(evidence.reason, factor);
    }
    return OutputMatches(context, b.output, ExpectedIdentitySolve2<Value>());
  }
  const std::array<rund::AccelGraphBufferRef, 4u> refs{
      rund::AccelRead(b.factor, "factor"), rund::AccelRead(b.rhs, "rhs"),
      rund::AccelWrite(b.output, "output"),
      rund::AccelWrite(b.status, "status")};
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelSolve(refs.data(), refs.size(), solve_desc),
  };
  const auto kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = nodes.data(),
                   .node_count = nodes.size(),
                   .scalar = scalar,
                   .domain = rund::kernel::ComputeDomain::Fixed,
                   .fixed_format = test::FixedFormatForLane(scalar),
               });
  const std::array<rund::AccelRunBinding, 4u> binds{
      rund::AccelRunBinding{.buffer = &b.factor,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &b.rhs,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &b.output,
                            .role = rund::kernel::BufferRole::Write},
      rund::AccelRunBinding{.buffer = &b.status,
                            .role = rund::kernel::BufferRole::Write},
  };
  const auto evidence =
      rund::node::accel::RunAccelKernel(context, kernel,
                                        rund::AccelRun{
                                            .bindings = binds.data(),
                                            .binding_count = binds.size(),
                                            .tile_count = 2u,
                                            .fresh_evidence = true,
                                        });
  constexpr std::uint64_t expected_dispatches = 1u;
  if (!evidence.ok || evidence.dispatch_count != expected_dispatches ||
      evidence.failed_batches != 0u ||
      !rund::node::accel::DownloadAccelBuffer(context, b.status, status.data(),
                                              sizeof(rund::kernel::u32))
           .ok ||
      status[0] !=
          static_cast<rund::kernel::u32>(rund::kernel::SolveStatus::Ok)) {
    return SolveReuseFail<Value>(evidence.reason, factor);
  }
  return OutputMatches(context, b.output, ExpectedIdentitySolve2<Value>());
}

} // namespace

[[nodiscard]] bool BackendRunsFactorReuseSolve(const rund::AccelDevice &pick) {
  return RunFactorReuseSolve<rund::kernel::i32>(pick,
                                                rund::kernel::FactorOp::LU) &&
         RunFactorReuseSolve<rund::kernel::i32>(pick,
                                                rund::kernel::FactorOp::QR) &&
         RunFactorReuseSolve<rund::kernel::i32>(
             pick, rund::kernel::FactorOp::Cholesky) &&
         RunFactorReuseSolve<rund::kernel::i64>(pick,
                                                rund::kernel::FactorOp::LU) &&
         RunFactorReuseSolve<rund::kernel::i64>(pick,
                                                rund::kernel::FactorOp::QR) &&
         RunFactorReuseSolve<rund::kernel::i64>(
             pick, rund::kernel::FactorOp::Cholesky);
}

} // namespace node_accel_contract
