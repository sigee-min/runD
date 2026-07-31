#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>

#include <accel/graph/factory/buffer/read.hpp>
#include <accel/graph/factory/buffer/write.hpp>
#include <accel/graph/factory/primitive/factor/node.hpp>
#include <accel/graph/factory/primitive/solve/node.hpp>
#include <accel/graph/factory/primitive/spectrum/node.hpp>
#include <kernel/program/compute/factor/reference.hpp>

#include "primitive/local.hpp"
#include "test/compute/fixed.hpp"
#include <node/accel/context.hpp>
#include <node/accel/pick.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <string_view>

namespace node_accel_contract {
namespace {

constexpr rund::kernel::i32 kOne = 0x40000000;
constexpr rund::kernel::i32 kVectorOne = 0x7fffffff;
constexpr rund::kernel::i64 kWideOne = 0x4000000000000000ll;
constexpr rund::kernel::i64 kWideVectorOne = 0x7fffffffffffffffll;

template <typename Value>
[[nodiscard]] bool FixedClose(const Value actual, const Value expected) {
  const long double scale = sizeof(Value) == sizeof(rund::kernel::i64)
                                ? 9223372036854775808.0L
                                : 2147483648.0L;
  return std::fabs((static_cast<long double>(actual) -
                    static_cast<long double>(expected)) /
                   scale) <= 1.0e-6L;
}

template <typename Value>
[[nodiscard]] bool FactorOutputMatches(const rund::AccelContext &context,
                                       const rund::AccelBuffer &factor,
                                       const rund::kernel::FactorOp op,
                                       const std::array<Value, 4u> &input) {
  std::array<Value, 4u> out{};
  if (!rund::node::accel::DownloadAccelBuffer(context, factor, out.data(),
                                              out.size() * sizeof(Value))
           .ok) {
    return false;
  }
  const Value one = static_cast<Value>(
      sizeof(Value) == sizeof(rund::kernel::i64) ? kWideVectorOne : kVectorOne);
  if (op == rund::kernel::FactorOp::LU) {
    for (std::size_t i = 0u; i < out.size(); ++i) {
      if (out[i] != input[i]) {
        return false;
      }
    }
    return true;
  }
  if (op == rund::kernel::FactorOp::QR) {
    return FixedClose(out[0], one) && out[1] == Value{0} &&
           out[2] == Value{0} && FixedClose(out[3], one);
  }
  return out[0] > Value{0} && out[1] == Value{0} && out[2] == Value{0} &&
         out[3] > Value{0};
}

template <typename Value, std::size_t Rows, std::size_t Batches>
[[nodiscard]] bool
FactorReferenceMatches(const rund::AccelContext &context,
                       const rund::AccelBuffer &factor,
                       const std::array<Value, Rows * Rows * Batches> &input,
                       const rund::kernel::FactorPlan &plan) {
  std::array<Value, Rows * Rows * Batches> expected{};
  std::array<Value, Rows * Rows * Batches> out{};
  std::array<rund::kernel::u32, Rows * Batches> aux{};
  std::array<rund::kernel::u32, Batches> status{};
  const rund::kernel::FactorResult reference = [&] {
    if constexpr (sizeof(Value) == sizeof(rund::kernel::i64)) {
      return rund::kernel::ReferenceFactorI64(
          input.data(), expected.data(), aux.data(), status.data(), plan);
    } else {
      return rund::kernel::ReferenceFactorI32(
          input.data(), expected.data(), aux.data(), status.data(), plan);
    }
  }();
  if (!rund::node::accel::DownloadAccelBuffer(context, factor, out.data(),
                                              out.size() * sizeof(Value))
           .ok ||
      !reference.ok || reference.failed_batches != 0u) {
    return false;
  }
  for (std::size_t index = 0u; index < out.size(); ++index) {
    if (out[index] != expected[index]) {
        return false;
    }
  }
  return true;
}

template <typename Value>
[[nodiscard]] bool RunFactor(const rund::AccelDevice &pick,
                             const rund::kernel::FactorOp op,
                             const std::array<Value, 4u> &input,
                             rund::kernel::FactorStatus expected_status) {
  namespace fix = node_accel_contract::primitive;
  const rund::kernel::ComputeScalar scalar =
      sizeof(Value) == sizeof(rund::kernel::i64)
          ? rund::kernel::ComputeScalar::Lane64
          : rund::kernel::ComputeScalar::Lane32;
  auto context = rund::node::accel::OpenAccel(pick);
  if (!context.check.ok) {
    return false;
  }
  auto in = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(Value),
                               input.size()));
  auto factor = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::WriteOnly, sizeof(Value),
                               input.size()));
  auto aux = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::WriteOnly,
                               sizeof(rund::kernel::u32), 2u));
  auto status = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::WriteOnly,
                               sizeof(rund::kernel::u32), 1u));
  if (!in.check.ok || !factor.check.ok || !status.check.ok ||
      !rund::node::accel::UploadAccelBuffer(context, in, input.data(),
                                            input.size() * sizeof(Value))
           .ok) {
    return false;
  }
  const bool lu = op == rund::kernel::FactorOp::LU;
  const std::array<rund::AccelGraphBufferRef, 4u> lu_refs{
      rund::AccelRead(in, "matrix"),
      rund::AccelWrite(factor, "factor"),
      rund::AccelWrite(aux, "aux"),
      rund::AccelWrite(status, "status"),
  };
  const std::array<rund::AccelGraphBufferRef, 3u> no_aux_refs{
      rund::AccelRead(in, "matrix"),
      rund::AccelWrite(factor, "factor"),
      rund::AccelWrite(status, "status"),
  };
  const rund::kernel::FactorDesc desc{
      .op = op,
      .layout = rund::kernel::MatrixLayout::RowMajor,
      .output = rund::kernel::FactorOutput::Packed,
      .pivot = (op == rund::kernel::FactorOp::QR || op == rund::kernel::FactorOp::Cholesky)
                   ? rund::kernel::PivotOp::None
                   : rund::kernel::PivotOp::Partial,
      .rows = 2u,
      .cols = 2u,
      .batch_count = 1u,
      .element_bytes = sizeof(Value),
      .fixed_format = test::FixedFormatForLane(
          scalar, rund::kernel::ComputeApproximation::Deterministic),
  };
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelFactor(lu ? lu_refs.data() : no_aux_refs.data(), lu ? 4u : 3u,
                        desc),
  };
  const auto kernel = rund::node::accel::CompileAccelKernel(
      context,
      rund::AccelGraph{.nodes = nodes.data(),
                       .node_count = nodes.size(),
                       .scalar = scalar,
                       .domain = rund::kernel::ComputeDomain::Fixed,
                       .fixed_format = test::FixedFormatForLane(scalar),
});
  if (!kernel.check.ok) {
    return false;
  }
  const std::array<rund::AccelRunBinding, 4u> lu_bindings{
      rund::AccelRunBinding{.buffer = &in,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &factor,
                            .role = rund::kernel::BufferRole::Write},
      rund::AccelRunBinding{.buffer = &aux,
                            .role = rund::kernel::BufferRole::Write},
      rund::AccelRunBinding{.buffer = &status,
                            .role = rund::kernel::BufferRole::Write},
  };
  const std::array<rund::AccelRunBinding, 3u> no_aux_bindings{
      rund::AccelRunBinding{.buffer = &in,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &factor,
                            .role = rund::kernel::BufferRole::Write},
      rund::AccelRunBinding{.buffer = &status,
                            .role = rund::kernel::BufferRole::Write},
  };
  const auto evidence = rund::node::accel::RunAccelKernel(
      context, kernel,
      rund::AccelRun{.bindings =
                         lu ? lu_bindings.data() : no_aux_bindings.data(),
                     .binding_count = lu ? 4u : 3u,
                     .tile_count = input.size(),
                     .fresh_evidence = true,
});
  std::array<rund::kernel::u32, 1u> status_out{};
  constexpr std::uint64_t expected_dispatches = 1u;
  if (!evidence.ok || evidence.dispatch_count != expected_dispatches ||
      evidence.host_to_device_bytes != 0u ||
      evidence.device_to_host_bytes != 0u ||
      !rund::node::accel::DownloadAccelBuffer(
           context, status, status_out.data(), sizeof(rund::kernel::u32))
           .ok) {
    return false;
  }
  const bool failed = expected_status != rund::kernel::FactorStatus::Ok;
  if (status_out[0] != static_cast<rund::kernel::u32>(expected_status) ||
      evidence.failed_batches != (failed ? 1u : 0u) ||
      (failed && evidence.first_status != status_out[0])) {
    return false;
  }
  if (!failed && !FactorOutputMatches(context, factor, op, input)) {
    std::array<Value, 4u> diagnostic{};
    const auto diagnostic_download = rund::node::accel::DownloadAccelBuffer(
        context, factor, diagnostic.data(), diagnostic.size() * sizeof(Value));
    std::cerr << "factor output mismatch device="
              << pick.backend_info.device_name << " op=" << static_cast<int>(op)
              << " bytes=" << sizeof(Value)
              << " download=" << diagnostic_download.ok << " values=";
    for (const Value value : diagnostic) {
      std::cerr << static_cast<long long>(value) << ',';
    }
    std::cerr << '\n';
    return false;
  }
  return true;
}

template <typename Value, std::size_t Rows>
[[nodiscard]] bool RunFactorDense(const rund::AccelDevice &pick,
                                  const rund::kernel::FactorOp op) {
  namespace fix = node_accel_contract::primitive;
  const rund::kernel::ComputeScalar scalar =
      sizeof(Value) == sizeof(rund::kernel::i64)
          ? rund::kernel::ComputeScalar::Lane64
          : rund::kernel::ComputeScalar::Lane32;
  auto context = rund::node::accel::OpenAccel(pick);
  if (!context.check.ok) {
    return false;
  }
  constexpr std::size_t batches = 3u;
  constexpr std::size_t count = Rows * Rows * batches;
  std::array<Value, count> input{};
  const Value diag = static_cast<Value>(
      sizeof(Value) == sizeof(rund::kernel::i64) ? kWideOne : kOne);
  for (std::size_t batch = 0u; batch < batches; ++batch) {
    const Value offdiag = static_cast<Value>(diag / (64 + batch * 16));
    for (std::size_t row = 0u; row < Rows; ++row) {
      for (std::size_t col = 0u; col < Rows; ++col) {
        input[batch * Rows * Rows + row * Rows + col] =
            row == col ? diag : offdiag;
      }
    }
  }
  auto in = rund::node::accel::CreateAccelBuffer(
      context,
      fix::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(Value), count));
  auto factor = rund::node::accel::CreateAccelBuffer(
      context,
      fix::BufferDesc(rund::BufferUsage::WriteOnly, sizeof(Value), count));
  auto aux = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::WriteOnly,
                               sizeof(rund::kernel::u32), Rows * batches));
  auto status = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::WriteOnly,
                               sizeof(rund::kernel::u32), batches));
  if (!in.check.ok || !factor.check.ok || !aux.check.ok || !status.check.ok ||
      !rund::node::accel::UploadAccelBuffer(context, in, input.data(),
                                            input.size() * sizeof(Value))
           .ok) {
    return false;
  }
  const bool lu = op == rund::kernel::FactorOp::LU;
  const std::array<rund::AccelGraphBufferRef, 4u> lu_refs{
      rund::AccelRead(in, "matrix"), rund::AccelWrite(factor, "factor"),
      rund::AccelWrite(aux, "aux"), rund::AccelWrite(status, "status")};
  const std::array<rund::AccelGraphBufferRef, 3u> no_aux_refs{
      rund::AccelRead(in, "matrix"), rund::AccelWrite(factor, "factor"),
      rund::AccelWrite(status, "status")};
  const rund::kernel::FactorDesc desc{
      .op = op,
      .layout = rund::kernel::MatrixLayout::RowMajor,
      .output = rund::kernel::FactorOutput::Packed,
      .pivot = (op == rund::kernel::FactorOp::QR || op == rund::kernel::FactorOp::Cholesky)
                   ? rund::kernel::PivotOp::None
                   : rund::kernel::PivotOp::Partial,
      .rows = Rows,
      .cols = Rows,
      .batch_count = batches,
      .element_bytes = sizeof(Value),
      .fixed_format = test::FixedFormatForLane(
          scalar, rund::kernel::ComputeApproximation::Deterministic),
  };
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelFactor(lu ? lu_refs.data() : no_aux_refs.data(), lu ? 4u : 3u,
                        desc),
  };
  const auto kernel = rund::node::accel::CompileAccelKernel(
      context,
      rund::AccelGraph{.nodes = nodes.data(),
                       .node_count = nodes.size(),
                       .scalar = scalar,
                       .domain = rund::kernel::ComputeDomain::Fixed,
                       .fixed_format = test::FixedFormatForLane(scalar),
});
  if (!kernel.check.ok) {
    return false;
  }
  const std::array<rund::AccelRunBinding, 4u> lu_bindings{
      rund::AccelRunBinding{.buffer = &in,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &factor,
                            .role = rund::kernel::BufferRole::Write},
      rund::AccelRunBinding{.buffer = &aux,
                            .role = rund::kernel::BufferRole::Write},
      rund::AccelRunBinding{.buffer = &status,
                            .role = rund::kernel::BufferRole::Write},
  };
  const std::array<rund::AccelRunBinding, 3u> no_aux_bindings{
      rund::AccelRunBinding{.buffer = &in,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &factor,
                            .role = rund::kernel::BufferRole::Write},
      rund::AccelRunBinding{.buffer = &status,
                            .role = rund::kernel::BufferRole::Write},
  };
  const auto evidence = rund::node::accel::RunAccelKernel(
      context, kernel,
      rund::AccelRun{.bindings =
                         lu ? lu_bindings.data() : no_aux_bindings.data(),
                     .binding_count = lu ? 4u : 3u,
                     .tile_count = Rows,
                     .fresh_evidence = true,
});
  std::array<rund::kernel::u32, batches> status_out{};
  constexpr std::uint64_t expected_dispatches = 1u;
  const rund::kernel::FactorPlan plan = rund::kernel::PlanFactor(desc);
  bool statuses_ok = rund::node::accel::DownloadAccelBuffer(
                         context, status, status_out.data(),
                         status_out.size() * sizeof(rund::kernel::u32))
                         .ok;
  for (const rund::kernel::u32 value : status_out) {
    statuses_ok = statuses_ok &&
                  value == static_cast<rund::kernel::u32>(
                               rund::kernel::FactorStatus::Ok);
  }
  return evidence.ok && evidence.dispatch_count == expected_dispatches &&
         evidence.host_to_device_bytes == 0u &&
         evidence.device_to_host_bytes == 0u && evidence.failed_batches == 0u &&
         statuses_ok && plan.ok &&
         FactorReferenceMatches<Value, Rows, batches>(context, factor, input,
                                                       plan);
}

} // namespace

[[nodiscard]] bool
NumericAlgebraRejectsOversizeShape(const rund::AccelDevice &pick) {
  auto context = rund::node::accel::OpenAccel(pick);
  if (!context.check.ok) {
    return false;
  }
  const std::array<rund::AccelGraphNode, 1u> factor{
      rund::AccelFactor(nullptr, 0u,
                        rund::kernel::FactorDesc{.op = rund::kernel::FactorOp::QR,
                                         .pivot = rund::kernel::PivotOp::None,
                                         .rows = 17u,
                                         .cols = 17u}),
  };
  const std::array<rund::AccelGraphNode, 1u> solve{
      rund::AccelSolve(nullptr, 0u,
                       rund::kernel::SolveDesc{.op = rund::kernel::SolveOp::Linear,
                                       .input = rund::kernel::SolveInput::Matrix,
                                       .factor = rund::kernel::FactorOp::LU,
                                       .rows = 17u,
                                       .rhs_cols = 1u}),
  };
  const std::array<rund::AccelGraphNode, 1u> spectrum{
      rund::AccelSpectrum(
          nullptr, 0u,
          rund::kernel::SpectrumDesc{.op = rund::kernel::SpectrumOp::Eigen,
                             .domain = rund::kernel::SpectrumDomain::SymmetricReal,
                             .vectors = rund::kernel::SpectrumVectors::ValuesOnly,
                             .rows = 17u,
                             .cols = 17u,
                             .max_iterations = 32u}),
  };
  const auto rejects = [&](const auto &nodes) {
    const auto kernel = rund::node::accel::CompileAccelKernel(
        context, rund::AccelGraph{.nodes = nodes.data(),
                                  .node_count = nodes.size(),
                                  .scalar = rund::kernel::ComputeScalar::Lane32,
                                  .domain = rund::kernel::ComputeDomain::Fixed,
                                  .fixed_format = test::FixedFormatForLane(
                                      rund::kernel::ComputeScalar::Lane32),
});
    return !kernel.check.ok && std::string_view{kernel.check.reason} ==
                                   "accel_kernel_graph_invalid";
  };
  return rejects(factor) && rejects(solve) && rejects(spectrum);
}

[[nodiscard]] bool BackendRunsFactor(const rund::AccelDevice &pick) {
  return RunFactor(pick, rund::kernel::FactorOp::LU,
                   std::array<rund::kernel::i32, 4u>{kOne, 0, 0, kOne},
                   rund::kernel::FactorStatus::Ok) &&
         RunFactor(pick, rund::kernel::FactorOp::QR,
                   std::array<rund::kernel::i32, 4u>{kOne, 0, 0, kOne},
                   rund::kernel::FactorStatus::Ok) &&
         RunFactorDense<rund::kernel::i32, 9u>(pick, rund::kernel::FactorOp::LU) &&
         RunFactorDense<rund::kernel::i32, 9u>(pick, rund::kernel::FactorOp::QR) &&
         RunFactorDense<rund::kernel::i32, 9u>(
             pick, rund::kernel::FactorOp::Cholesky) &&
         RunFactor(pick, rund::kernel::FactorOp::Cholesky,
                   std::array<rund::kernel::i32, 4u>{0, kOne, kOne, 0},
                   rund::kernel::FactorStatus::NonSpd) &&
         RunFactor(pick, rund::kernel::FactorOp::Cholesky,
                   std::array<rund::kernel::i32, 4u>{kOne, 0, 0, kOne},
                   rund::kernel::FactorStatus::Ok) &&
         RunFactor(pick, rund::kernel::FactorOp::LU,
                   std::array<rund::kernel::i64, 4u>{kWideOne, 0, 0, kWideOne},
                   rund::kernel::FactorStatus::Ok) &&
         RunFactor(pick, rund::kernel::FactorOp::QR,
                   std::array<rund::kernel::i64, 4u>{kWideOne, 0, 0, kWideOne},
                   rund::kernel::FactorStatus::Ok) &&
         RunFactorDense<rund::kernel::i64, 9u>(pick, rund::kernel::FactorOp::LU) &&
         RunFactorDense<rund::kernel::i64, 9u>(pick, rund::kernel::FactorOp::QR) &&
         RunFactorDense<rund::kernel::i64, 9u>(
             pick, rund::kernel::FactorOp::Cholesky) &&
         RunFactor(pick, rund::kernel::FactorOp::Cholesky,
                   std::array<rund::kernel::i64, 4u>{kWideOne, 0, 0, kWideOne},
                   rund::kernel::FactorStatus::Ok);
}

[[nodiscard]] bool AvailableBackendsRunFactorNatively() {
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
    if (!BackendRunsFactor(pick)) {
      return false;
    }
  }
  return true;
}

} // namespace node_accel_contract
