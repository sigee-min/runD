#include "model.hpp"

namespace node_accel_contract {
namespace {

using namespace solve_contract;

template <typename Value>
[[nodiscard]] bool
RunRawSolve(const rund::AccelDevice &pick, const std::array<Value, 4u> &matrix,
            rund::kernel::SolveStatus expected_status,
            const rund::kernel::FactorOp factor = rund::kernel::FactorOp::LU) {
  namespace fix = node_accel_contract::primitive;
  const rund::kernel::ComputeScalar scalar =
      sizeof(Value) == sizeof(rund::kernel::i64)
          ? rund::kernel::ComputeScalar::Lane64
          : rund::kernel::ComputeScalar::Lane32;
  auto context = rund::node::accel::OpenAccel(pick);
  SolveBuffers b{};
  if (!context.check.ok || !CreateSolveBuffers<Value>(context, b) ||
      !rund::node::accel::UploadAccelBuffer(context, b.matrix, matrix.data(),
                                            matrix.size() * sizeof(Value))
           .ok) {
    return SolveFail("raw.buffer");
  }
  const std::array<Value, 2u> rhs{
      static_cast<Value>(sizeof(Value) == sizeof(rund::kernel::i64) ? kWideHalf
                                                                    : kHalf),
      static_cast<Value>(sizeof(Value) == sizeof(rund::kernel::i64)
                             ? kWideQuarter
                             : kQuarter)};
  if (!rund::node::accel::UploadAccelBuffer(context, b.rhs, rhs.data(),
                                            rhs.size() * sizeof(Value))
           .ok) {
    return SolveFail("raw.rhs");
  }
  const std::array<rund::AccelGraphBufferRef, 4u> refs{
      rund::AccelRead(b.matrix, "matrix"), rund::AccelRead(b.rhs, "rhs"),
      rund::AccelWrite(b.output, "output"),
      rund::AccelWrite(b.status, "status")};
  const rund::kernel::SolveDesc desc{
      .op = rund::kernel::SolveOp::Linear,
      .input = rund::kernel::SolveInput::Matrix,
      .factor = factor,
      .layout = rund::kernel::MatrixLayout::RowMajor,
      .pivot = factor == rund::kernel::FactorOp::LU
                   ? rund::kernel::PivotOp::Partial
                   : rund::kernel::PivotOp::None,
      .rows = 2u,
      .rhs_cols = 1u,
      .element_bytes = sizeof(Value),
      .fixed_format = FixedFormat(scalar),
  };
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelSolve(refs.data(), refs.size(), desc),
  };
  const auto kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = nodes.data(),
                   .node_count = nodes.size(),
                   .scalar = scalar,
                   .domain = rund::kernel::ComputeDomain::Fixed,
                   .fixed_format = test::FixedFormatForLane(scalar),
               });
  if (!kernel.check.ok) {
    return SolveFail(kernel.check.reason);
  }
  const std::array<rund::AccelRunBinding, 4u> bindings{
      rund::AccelRunBinding{.buffer = &b.matrix,
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
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = 2u,
                                            .fresh_evidence = true,
                                        });
  std::array<rund::kernel::u32, 1u> status{};
  constexpr std::uint64_t expected_dispatches = 1u;
  if (!evidence.ok || evidence.dispatch_count != expected_dispatches ||
      !rund::node::accel::DownloadAccelBuffer(context, b.status, status.data(),
                                              sizeof(rund::kernel::u32))
           .ok) {
    return SolveFail(evidence.reason);
  }
  const bool failed = expected_status != rund::kernel::SolveStatus::Ok;
  if (status[0] != static_cast<rund::kernel::u32>(expected_status) ||
      evidence.failed_batches != (failed ? 1u : 0u)) {
    return SolveFail("raw.status");
  }
  if (failed) {
    return true;
  }
  if (!OutputMatches(context, b.output, ExpectedIdentitySolve2<Value>())) {
    return SolveFail("raw.output");
  }
  return true;
}

template <typename Value>
[[nodiscard]] bool RunRawSolve3(const rund::AccelDevice &pick) {
  namespace fix = node_accel_contract::primitive;
  const rund::kernel::ComputeScalar scalar =
      sizeof(Value) == sizeof(rund::kernel::i64)
          ? rund::kernel::ComputeScalar::Lane64
          : rund::kernel::ComputeScalar::Lane32;
  auto context = rund::node::accel::OpenAccel(pick);
  SolveBuffers b{};
  if (!context.check.ok || !CreateSolveBuffers<Value>(context, b, 9u, 9u, 3u)) {
    return SolveFail("raw3.buffer");
  }
  const Value one = static_cast<Value>(
      sizeof(Value) == sizeof(rund::kernel::i64) ? kWideOne : kOne);
  const std::array<Value, 9u> matrix{one, 0, 0, 0, one, 0, 0, 0, one};
  const std::array<Value, 3u> rhs{
      static_cast<Value>(sizeof(Value) == sizeof(rund::kernel::i64) ? kWideHalf
                                                                    : kHalf),
      static_cast<Value>(
          sizeof(Value) == sizeof(rund::kernel::i64) ? kWideQuarter : kQuarter),
      static_cast<Value>(sizeof(Value) == sizeof(rund::kernel::i64) ? kWideHalf
                                                                    : kHalf)};
  if (!rund::node::accel::UploadAccelBuffer(context, b.matrix, matrix.data(),
                                            matrix.size() * sizeof(Value))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(context, b.rhs, rhs.data(),
                                            rhs.size() * sizeof(Value))
           .ok) {
    return SolveFail("raw3.upload");
  }
  const std::array<rund::AccelGraphBufferRef, 4u> refs{
      rund::AccelRead(b.matrix, "matrix"), rund::AccelRead(b.rhs, "rhs"),
      rund::AccelWrite(b.output, "output"),
      rund::AccelWrite(b.status, "status")};
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelSolve(refs.data(), refs.size(),
                       rund::kernel::SolveDesc{
                           .op = rund::kernel::SolveOp::Linear,
                           .input = rund::kernel::SolveInput::Matrix,
                           .factor = rund::kernel::FactorOp::LU,
                           .layout = rund::kernel::MatrixLayout::RowMajor,
                           .pivot = rund::kernel::PivotOp::Partial,
                           .rows = 3u,
                           .rhs_cols = 1u,
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
  if (!kernel.check.ok) {
    return SolveFail(kernel.check.reason);
  }
  const std::array<rund::AccelRunBinding, 4u> bindings{
      rund::AccelRunBinding{.buffer = &b.matrix,
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
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = 3u,
                                            .fresh_evidence = true,
                                        });
  std::array<rund::kernel::u32, 1u> status{};
  constexpr std::uint64_t expected_dispatches = 1u;
  return evidence.ok && evidence.dispatch_count == expected_dispatches &&
         evidence.failed_batches == 0u &&
         rund::node::accel::DownloadAccelBuffer(
             context, b.status, status.data(), sizeof(rund::kernel::u32))
             .ok &&
         status[0] ==
             static_cast<rund::kernel::u32>(rund::kernel::SolveStatus::Ok) &&
         (OutputMatches(context, b.output, ExpectedIdentitySolve3<Value>()) ||
          SolveFail("raw3.output"));
}

template <typename Value, std::size_t Rows>
[[nodiscard]] bool RunRawSolveDense(const rund::AccelDevice &pick,
                                    const rund::kernel::FactorOp factor) {
  namespace fix = node_accel_contract::primitive;
  const rund::kernel::ComputeScalar scalar =
      sizeof(Value) == sizeof(rund::kernel::i64)
          ? rund::kernel::ComputeScalar::Lane64
          : rund::kernel::ComputeScalar::Lane32;
  auto context = rund::node::accel::OpenAccel(pick);
  SolveBuffers b{};
  constexpr std::size_t batches = 3u;
  constexpr std::size_t rhs_cols = 3u;
  constexpr std::uint64_t matrix_count = Rows * Rows * batches;
  constexpr std::uint64_t rhs_count = Rows * rhs_cols * batches;
  if (!context.check.ok ||
      !CreateSolveBuffers<Value>(context, b, matrix_count, matrix_count,
                                 rhs_count, Rows * batches, batches)) {
    return SolveFail("rawn.buffer");
  }
  std::array<Value, matrix_count> matrix{};
  const Value diag = static_cast<Value>(
      sizeof(Value) == sizeof(rund::kernel::i64) ? kWideOne : kOne);
  for (std::size_t batch = 0u; batch < batches; ++batch) {
    const Value offdiag = static_cast<Value>(diag / (64 + batch * 16));
    for (std::size_t row = 0u; row < Rows; ++row) {
      for (std::size_t col = 0u; col < Rows; ++col) {
        matrix[batch * Rows * Rows + row * Rows + col] =
            row == col ? diag : offdiag;
      }
    }
  }
  std::array<Value, rhs_count> rhs{};
  for (std::size_t batch = 0u; batch < batches; ++batch) {
    for (std::size_t row = 0u; row < Rows; ++row) {
      for (std::size_t col = 0u; col < rhs_cols; ++col) {
        const Value base = static_cast<Value>(
            sizeof(Value) == sizeof(rund::kernel::i64) ? kWideQuarter
                                                       : kQuarter);
        rhs[batch * Rows * rhs_cols + row * rhs_cols + col] =
            static_cast<Value>(base / static_cast<Value>(1u + col) +
                               static_cast<Value>((row + batch) % 3u));
      }
    }
  }
  if (!rund::node::accel::UploadAccelBuffer(context, b.matrix, matrix.data(),
                                            matrix.size() * sizeof(Value))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(context, b.rhs, rhs.data(),
                                            rhs.size() * sizeof(Value))
           .ok) {
    return SolveFail("rawn.upload");
  }
  const std::array<rund::AccelGraphBufferRef, 4u> refs{
      rund::AccelRead(b.matrix, "matrix"), rund::AccelRead(b.rhs, "rhs"),
      rund::AccelWrite(b.output, "output"),
      rund::AccelWrite(b.status, "status")};
  const rund::kernel::SolveDesc desc{
      .op = rund::kernel::SolveOp::Linear,
      .input = rund::kernel::SolveInput::Matrix,
      .factor = factor,
      .layout = rund::kernel::MatrixLayout::RowMajor,
      .pivot = factor == rund::kernel::FactorOp::LU
                   ? rund::kernel::PivotOp::Partial
                   : rund::kernel::PivotOp::None,
      .rows = Rows,
      .rhs_cols = rhs_cols,
      .batch_count = batches,
      .element_bytes = sizeof(Value),
      .fixed_format = FixedFormat(scalar),
  };
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelSolve(refs.data(), refs.size(), desc),
  };
  const auto kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = nodes.data(),
                   .node_count = nodes.size(),
                   .scalar = scalar,
                   .domain = rund::kernel::ComputeDomain::Fixed,
                   .fixed_format = test::FixedFormatForLane(scalar),
               });
  if (!kernel.check.ok) {
    return SolveFail(kernel.check.reason);
  }
  const std::array<rund::AccelRunBinding, 4u> bindings{
      rund::AccelRunBinding{.buffer = &b.matrix,
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
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = Rows * batches,
                                            .fresh_evidence = true,
                                        });
  const rund::kernel::SolvePlan plan = rund::kernel::PlanSolve(desc);
  std::array<Value, rhs_count> expected{};
  std::array<rund::kernel::u32, Rows * batches> reference_aux{};
  std::array<rund::kernel::u32, batches> expected_status{};
  const rund::kernel::SolveResult reference = [&] {
    if constexpr (sizeof(Value) == sizeof(rund::kernel::i64)) {
      return rund::kernel::ReferenceSolveI64(
          matrix.data(), reference_aux.data(), rhs.data(), expected.data(),
          expected_status.data(), plan);
    } else {
      return rund::kernel::ReferenceSolveI32(
          matrix.data(), reference_aux.data(), rhs.data(), expected.data(),
          expected_status.data(), plan);
    }
  }();
  std::array<rund::kernel::u32, batches> status{};
  constexpr std::uint64_t expected_dispatches = 1u;
  if (!plan.ok || !reference.ok || reference.failed_batches != 0u ||
      !evidence.ok || evidence.dispatch_count != expected_dispatches ||
      evidence.failed_batches != 0u ||
      !rund::node::accel::DownloadAccelBuffer(context, b.status, status.data(),
                                              status.size() *
                                                  sizeof(rund::kernel::u32))
           .ok ||
      status != expected_status) {
    return SolveFail("rawn.status");
  }
  const bool output_matches =
      pick.api == rund::AccelApi::Vulkan
          ? ExactOutputMatches(context, b.output, expected)
          : OutputMatches(context, b.output, expected);
  if (!output_matches) {
    std::cerr << "solve dense mismatch device=" << pick.backend_info.device_name
              << " factor=" << static_cast<int>(factor)
              << " bytes=" << sizeof(Value) << '\n';
    return SolveFail("rawn.output");
  }
  return true;
}

} // namespace

[[nodiscard]] bool BackendRunsSolve(const rund::AccelDevice &pick) {
  return RunRawSolve(pick, std::array<rund::kernel::i32, 4u>{kOne, 0, 0, kOne},
                     rund::kernel::SolveStatus::Ok) &&
         RunRawSolve(pick, std::array<rund::kernel::i32, 4u>{kOne, 0, 0, kOne},
                     rund::kernel::SolveStatus::Ok,
                     rund::kernel::FactorOp::QR) &&
         RunRawSolve(pick, std::array<rund::kernel::i32, 4u>{kOne, 0, 0, kOne},
                     rund::kernel::SolveStatus::Ok,
                     rund::kernel::FactorOp::Cholesky) &&
         RunRawSolve(pick,
                     std::array<rund::kernel::i32, 4u>{kOne, kHalf, kHalf, 0},
                     rund::kernel::SolveStatus::NonSpd,
                     rund::kernel::FactorOp::Cholesky) &&
         RunRawSolve(pick, std::array<rund::kernel::i32, 4u>{0, 0, 0, 0},
                     rund::kernel::SolveStatus::Singular) &&
         RunRawSolve3<rund::kernel::i32>(pick) &&
         RunRawSolveDense<rund::kernel::i32, 9u>(pick,
                                                 rund::kernel::FactorOp::LU) &&
         RunRawSolveDense<rund::kernel::i32, 9u>(pick,
                                                 rund::kernel::FactorOp::QR) &&
         RunRawSolveDense<rund::kernel::i32, 9u>(
             pick, rund::kernel::FactorOp::Cholesky) &&
         RunRawSolve(
             pick, std::array<rund::kernel::i64, 4u>{kWideOne, 0, 0, kWideOne},
             rund::kernel::SolveStatus::Ok) &&
         RunRawSolve(
             pick, std::array<rund::kernel::i64, 4u>{kWideOne, 0, 0, kWideOne},
             rund::kernel::SolveStatus::Ok, rund::kernel::FactorOp::QR) &&
         RunRawSolve(
             pick, std::array<rund::kernel::i64, 4u>{kWideOne, 0, 0, kWideOne},
             rund::kernel::SolveStatus::Ok, rund::kernel::FactorOp::Cholesky) &&
         RunRawSolve(pick,
                     std::array<rund::kernel::i64, 4u>{kWideOne, kWideHalf,
                                                       kWideHalf, 0},
                     rund::kernel::SolveStatus::NonSpd,
                     rund::kernel::FactorOp::Cholesky) &&
         RunRawSolve(pick, std::array<rund::kernel::i64, 4u>{0, 0, 0, 0},
                     rund::kernel::SolveStatus::Singular) &&
         RunRawSolve3<rund::kernel::i64>(pick) &&
         RunRawSolveDense<rund::kernel::i64, 9u>(pick,
                                                 rund::kernel::FactorOp::LU) &&
         RunRawSolveDense<rund::kernel::i64, 9u>(pick,
                                                 rund::kernel::FactorOp::QR) &&
         RunRawSolveDense<rund::kernel::i64, 9u>(
             pick, rund::kernel::FactorOp::Cholesky);
}

} // namespace node_accel_contract
