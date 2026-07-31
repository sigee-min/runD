#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>

#include <accel/graph/factory/buffer/read.hpp>
#include <accel/graph/factory/buffer/write.hpp>
#include <accel/graph/factory/primitive/spectrum/node.hpp>
#include <kernel/program/compute/spectrum/reference.hpp>

#include "primitive/local.hpp"
#include "test/compute/fixed.hpp"
#include <node/accel/context.hpp>
#include <node/accel/pick.hpp>

#include <array>
#include <cstddef>
#include <iostream>

namespace node_accel_contract {
namespace {

constexpr rund::kernel::i32 kOne = 0x40000000;
constexpr rund::kernel::i32 kHalf = 0x20000000;
constexpr rund::kernel::i32 kVectorOne = 0x7fffffff;
constexpr rund::kernel::i64 kWideOne = 0x4000000000000000ll;
constexpr rund::kernel::i64 kWideHalf = 0x2000000000000000ll;
constexpr rund::kernel::i64 kWideVectorOne = 0x7fffffffffffffffll;

template <typename Value, std::size_t Rows>
[[nodiscard]] bool RunSpectrum(const rund::AccelDevice &pick,
                               const rund::kernel::SpectrumOp op) {
  namespace fix = node_accel_contract::primitive;
  const rund::kernel::ComputeScalar scalar =
      sizeof(Value) == sizeof(rund::kernel::i64)
          ? rund::kernel::ComputeScalar::Lane64
          : rund::kernel::ComputeScalar::Lane32;
  auto context = rund::node::accel::OpenAccel(pick);
  if (!context.check.ok) {
    return false;
  }
  auto input = rund::node::accel::CreateAccelBuffer(
      context,
      fix::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(Value), Rows * Rows));
  auto values = rund::node::accel::CreateAccelBuffer(
      context,
      fix::BufferDesc(rund::BufferUsage::WriteOnly, sizeof(Value), Rows));
  auto status = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::WriteOnly,
                               sizeof(rund::kernel::u32), 1u));
  std::array<Value, Rows * Rows> matrix{};
  for (std::size_t i = 0u; i < Rows; ++i) {
    matrix[i * Rows + i] = static_cast<Value>(
        i == 0u
            ? (sizeof(Value) == sizeof(rund::kernel::i64) ? kWideOne : kOne)
            : (sizeof(Value) == sizeof(rund::kernel::i64) ? kWideHalf : kHalf));
  }
  if (!input.check.ok || !values.check.ok || !status.check.ok ||
      !rund::node::accel::UploadAccelBuffer(context, input, matrix.data(),
                                            matrix.size() * sizeof(Value))
           .ok) {
    return false;
  }
  const std::array<rund::AccelGraphBufferRef, 3u> refs{
      rund::AccelRead(input, "matrix"), rund::AccelWrite(values, "values"),
      rund::AccelWrite(status, "status")};
  const rund::kernel::SpectrumDesc desc{
      .op = op,
      .domain = op == rund::kernel::SpectrumOp::Eigen
                    ? rund::kernel::SpectrumDomain::SymmetricReal
                    : rund::kernel::SpectrumDomain::GeneralReal,
      .vectors = rund::kernel::SpectrumVectors::ValuesOnly,
      .layout = rund::kernel::MatrixLayout::RowMajor,
      .rows = Rows,
      .cols = Rows,
      .max_iterations = 64u,
      .element_bytes = sizeof(Value),
      .fixed_format = test::FixedFormatForLane(
          scalar, rund::kernel::ComputeApproximation::Deterministic),
  };
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelSpectrum(refs.data(), refs.size(), desc),
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
  const std::array<rund::AccelRunBinding, 3u> bindings{
      rund::AccelRunBinding{.buffer = &input,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &values,
                            .role = rund::kernel::BufferRole::Write},
      rund::AccelRunBinding{.buffer = &status,
                            .role = rund::kernel::BufferRole::Write},
  };
  const auto evidence = rund::node::accel::RunAccelKernel(
      context, kernel,
      rund::AccelRun{.bindings = bindings.data(),
                     .binding_count = bindings.size(),
                     .tile_count = 2u,
                     .fresh_evidence = true,
});
  std::array<rund::kernel::u32, 1u> status_out{};
  std::array<Value, Rows> values_out{};
  const Value expected_first = static_cast<Value>(
      sizeof(Value) == sizeof(rund::kernel::i64) ? kWideOne : kOne);
  const Value expected_rest = static_cast<Value>(
      sizeof(Value) == sizeof(rund::kernel::i64) ? kWideHalf : kHalf);
  const bool values_ok =
      rund::node::accel::DownloadAccelBuffer(context, values, values_out.data(),
                                             values_out.size() * sizeof(Value))
          .ok;
  bool values_match = values_ok && values_out[0] == expected_first;
  for (std::size_t i = 1u; i < Rows; ++i) {
    values_match = values_match && values_out[i] == expected_rest;
  }
  constexpr std::uint64_t expected_dispatches = 1u;
  return evidence.ok && evidence.dispatch_count == expected_dispatches &&
         evidence.failed_batches == 0u &&
         rund::node::accel::DownloadAccelBuffer(
             context, status, status_out.data(), sizeof(rund::kernel::u32))
             .ok &&
         status_out[0] ==
             static_cast<rund::kernel::u32>(rund::kernel::SpectrumStatus::Ok) &&
         values_match;
}

template <typename Value, std::size_t Rows>
[[nodiscard]] bool RunSpectrumVectors(const rund::AccelDevice &pick,
                                      const rund::kernel::SpectrumOp op) {
  namespace fix = node_accel_contract::primitive;
  const rund::kernel::ComputeScalar scalar =
      sizeof(Value) == sizeof(rund::kernel::i64)
          ? rund::kernel::ComputeScalar::Lane64
          : rund::kernel::ComputeScalar::Lane32;
  auto context = rund::node::accel::OpenAccel(pick);
  if (!context.check.ok) {
    return false;
  }
  auto input = rund::node::accel::CreateAccelBuffer(
      context,
      fix::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(Value), Rows * Rows));
  auto values = rund::node::accel::CreateAccelBuffer(
      context,
      fix::BufferDesc(rund::BufferUsage::WriteOnly, sizeof(Value), Rows));
  auto vectors = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::WriteOnly, sizeof(Value),
                               Rows * Rows));
  auto status = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::WriteOnly,
                               sizeof(rund::kernel::u32), 1u));
  std::array<Value, Rows * Rows> matrix{};
  for (std::size_t i = 0u; i < Rows; ++i) {
    matrix[i * Rows + i] = static_cast<Value>(
        i == 0u
            ? (sizeof(Value) == sizeof(rund::kernel::i64) ? kWideOne : kOne)
            : (sizeof(Value) == sizeof(rund::kernel::i64) ? kWideHalf : kHalf));
  }
  if (!input.check.ok || !values.check.ok || !vectors.check.ok ||
      !status.check.ok ||
      !rund::node::accel::UploadAccelBuffer(context, input, matrix.data(),
                                            matrix.size() * sizeof(Value))
           .ok) {
    return false;
  }
  const std::array<rund::AccelGraphBufferRef, 4u> refs{
      rund::AccelRead(input, "matrix"), rund::AccelWrite(values, "values"),
      rund::AccelWrite(vectors, "vectors"), rund::AccelWrite(status, "status")};
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelSpectrum(
          refs.data(), refs.size(),
          rund::kernel::SpectrumDesc{
              .op = op,
              .domain = op == rund::kernel::SpectrumOp::Eigen
                            ? rund::kernel::SpectrumDomain::SymmetricReal
                            : rund::kernel::SpectrumDomain::GeneralReal,
              .vectors = rund::kernel::SpectrumVectors::Full,
              .rows = Rows,
              .cols = Rows,
              .max_iterations = 64u,
              .element_bytes = sizeof(Value),
              .fixed_format = rund::kernel::PrimitiveFixedFormat(
                  test::FixedFormatForLane(scalar),
                  rund::kernel::ComputeApproximation::Deterministic)}),
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
  const std::array<rund::AccelRunBinding, 4u> bindings{
      rund::AccelRunBinding{.buffer = &input,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &values,
                            .role = rund::kernel::BufferRole::Write},
      rund::AccelRunBinding{.buffer = &vectors,
                            .role = rund::kernel::BufferRole::Write},
      rund::AccelRunBinding{.buffer = &status,
                            .role = rund::kernel::BufferRole::Write},
  };
  const auto evidence = rund::node::accel::RunAccelKernel(
      context, kernel,
      rund::AccelRun{.bindings = bindings.data(),
                     .binding_count = bindings.size(),
                     .tile_count = Rows,
                     .fresh_evidence = true,
});
  std::array<rund::kernel::u32, 1u> status_out{};
  std::array<Value, Rows * Rows> vectors_out{};
  const Value one = static_cast<Value>(
      sizeof(Value) == sizeof(rund::kernel::i64) ? kWideVectorOne : kVectorOne);
  bool vectors_match = rund::node::accel::DownloadAccelBuffer(
                           context, vectors, vectors_out.data(),
                           vectors_out.size() * sizeof(Value))
                           .ok;
  for (std::size_t row = 0u; row < Rows; ++row) {
    for (std::size_t col = 0u; col < Rows; ++col) {
      vectors_match = vectors_match && vectors_out[row * Rows + col] ==
                                           (row == col ? one : Value{0});
    }
  }
  constexpr std::uint64_t expected_dispatches = 1u;
  return evidence.ok && evidence.dispatch_count == expected_dispatches &&
         evidence.failed_batches == 0u &&
         rund::node::accel::DownloadAccelBuffer(
             context, status, status_out.data(), sizeof(rund::kernel::u32))
             .ok &&
         status_out[0] ==
             static_cast<rund::kernel::u32>(rund::kernel::SpectrumStatus::Ok) &&
         vectors_match;
}

template <typename Value, std::size_t Rows>
[[nodiscard]] bool RunSpectrumDense(const rund::AccelDevice &pick,
                                    const rund::kernel::SpectrumOp op) {
  namespace fix = node_accel_contract::primitive;
  constexpr std::size_t batches = 3u;
  constexpr std::size_t input_count = Rows * Rows * batches;
  constexpr std::size_t value_count = Rows * batches;
  const rund::kernel::ComputeScalar scalar =
      sizeof(Value) == sizeof(rund::kernel::i64)
          ? rund::kernel::ComputeScalar::Lane64
          : rund::kernel::ComputeScalar::Lane32;
  auto context = rund::node::accel::OpenAccel(pick);
  if (!context.check.ok) {
    return false;
  }
  auto input = rund::node::accel::CreateAccelBuffer(
      context,
      fix::BufferDesc(rund::BufferUsage::ReadOnly, sizeof(Value), input_count));
  auto values = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::WriteOnly, sizeof(Value),
                               value_count));
  auto status = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::WriteOnly,
                               sizeof(rund::kernel::u32), batches));
  std::array<Value, input_count> matrix{};
  const Value quarter = static_cast<Value>(
      sizeof(Value) == sizeof(rund::kernel::i64) ? kWideHalf : kHalf);
  for (std::size_t batch = 0u; batch < batches; ++batch) {
    for (std::size_t row = 0u; row < Rows; ++row) {
      for (std::size_t col = 0u; col < Rows; ++col) {
        matrix[batch * Rows * Rows + row * Rows + col] =
            row == col
                ? static_cast<Value>(quarter +
                                     quarter /
                                         static_cast<Value>(8u + row + batch))
                : ((row / 2u == col / 2u)
                       ? static_cast<Value>(
                             quarter /
                             static_cast<Value>(32u + batch * 4u))
                       : Value{1});
      }
    }
  }
  if (!input.check.ok || !values.check.ok || !status.check.ok ||
      !rund::node::accel::UploadAccelBuffer(context, input, matrix.data(),
                                            matrix.size() * sizeof(Value))
           .ok) {
    return false;
  }
  const std::array<rund::AccelGraphBufferRef, 3u> refs{
      rund::AccelRead(input, "matrix"), rund::AccelWrite(values, "values"),
      rund::AccelWrite(status, "status")};
  const rund::kernel::SpectrumDesc desc{
      .op = op,
      .domain = op == rund::kernel::SpectrumOp::Eigen
                    ? rund::kernel::SpectrumDomain::SymmetricReal
                    : rund::kernel::SpectrumDomain::GeneralReal,
      .vectors = rund::kernel::SpectrumVectors::ValuesOnly,
      .layout = rund::kernel::MatrixLayout::RowMajor,
      .rows = Rows,
      .cols = Rows,
      .batch_count = batches,
      .max_iterations = 256u,
      .element_bytes = sizeof(Value),
      .fixed_format = test::FixedFormatForLane(
          scalar, rund::kernel::ComputeApproximation::Deterministic),
  };
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelSpectrum(refs.data(), refs.size(), desc),
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
  const std::array<rund::AccelRunBinding, 3u> bindings{
      rund::AccelRunBinding{.buffer = &input,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &values,
                            .role = rund::kernel::BufferRole::Write},
      rund::AccelRunBinding{.buffer = &status,
                            .role = rund::kernel::BufferRole::Write},
  };
  const auto evidence = rund::node::accel::RunAccelKernel(
      context, kernel,
      rund::AccelRun{.bindings = bindings.data(),
                     .binding_count = bindings.size(),
                     .tile_count = Rows * batches,
                     .fresh_evidence = true,
});
  const rund::kernel::SpectrumPlan plan = rund::kernel::PlanSpectrum(desc);
  std::array<Value, value_count> expected{};
  std::array<rund::kernel::u32, batches> expected_status{};
  const rund::kernel::SpectrumResult reference = [&] {
    if constexpr (sizeof(Value) == sizeof(rund::kernel::i64)) {
      return rund::kernel::ReferenceSpectrumI64(
          matrix.data(), expected.data(), nullptr, expected_status.data(), plan);
    } else {
      return rund::kernel::ReferenceSpectrumI32(
          matrix.data(), expected.data(), nullptr, expected_status.data(), plan);
    }
  }();
  std::array<Value, value_count> actual{};
  std::array<rund::kernel::u32, batches> actual_status{};
  const bool downloaded =
      rund::node::accel::DownloadAccelBuffer(
          context, values, actual.data(), actual.size() * sizeof(Value))
          .ok &&
      rund::node::accel::DownloadAccelBuffer(
          context, status, actual_status.data(),
          actual_status.size() * sizeof(rund::kernel::u32))
          .ok;
  if (!plan.ok || !reference.ok || reference.failed_batches != 0u ||
      !evidence.ok || evidence.dispatch_count != 1u ||
      evidence.failed_batches != 0u || !downloaded ||
      actual_status != expected_status || actual != expected) {
    std::cerr << "spectrum dense mismatch device="
              << pick.backend_info.device_name << " op="
              << static_cast<int>(op) << " bytes=" << sizeof(Value)
              << " plan=" << plan.ok << " reference=" << reference.ok
              << " reference_failed=" << reference.failed_batches
              << " evidence=" << evidence.ok
              << " evidence_failed=" << evidence.failed_batches
              << " downloaded=" << downloaded << '\n';
    for (std::size_t index = 0u; index < batches; ++index) {
      std::cerr << "status[" << index << "]=" << actual_status[index]
                << " expected=" << expected_status[index] << '\n';
    }
    for (std::size_t index = 0u; index < value_count; ++index) {
      if (actual[index] != expected[index]) {
        std::cerr << "value[" << index << "]=" << actual[index]
                  << " expected=" << expected[index] << '\n';
        break;
      }
    }
    return false;
  }
  return true;
}

} // namespace

[[nodiscard]] bool BackendRunsSpectrum(const rund::AccelDevice &pick) {
  const bool dense =
      pick.api != rund::AccelApi::Vulkan ||
      (RunSpectrumDense<rund::kernel::i32, 9u>(
           pick, rund::kernel::SpectrumOp::Eigen) &&
       RunSpectrumDense<rund::kernel::i32, 9u>(
           pick, rund::kernel::SpectrumOp::SVD) &&
       RunSpectrumDense<rund::kernel::i64, 9u>(
           pick, rund::kernel::SpectrumOp::Eigen) &&
       RunSpectrumDense<rund::kernel::i64, 9u>(
           pick, rund::kernel::SpectrumOp::SVD));
  return dense &&
         RunSpectrum<rund::kernel::i32, 2u>(pick, rund::kernel::SpectrumOp::Eigen) &&
         RunSpectrum<rund::kernel::i32, 3u>(pick, rund::kernel::SpectrumOp::Eigen) &&
         RunSpectrum<rund::kernel::i32, 9u>(pick, rund::kernel::SpectrumOp::Eigen) &&
         RunSpectrum<rund::kernel::i32, 2u>(pick, rund::kernel::SpectrumOp::SVD) &&
         RunSpectrum<rund::kernel::i32, 3u>(pick, rund::kernel::SpectrumOp::SVD) &&
         RunSpectrum<rund::kernel::i32, 9u>(pick, rund::kernel::SpectrumOp::SVD) &&
         RunSpectrumVectors<rund::kernel::i32, 3u>(pick,
                                                   rund::kernel::SpectrumOp::Eigen) &&
         RunSpectrumVectors<rund::kernel::i32, 3u>(pick,
                                                   rund::kernel::SpectrumOp::SVD) &&
         RunSpectrum<rund::kernel::i64, 2u>(pick, rund::kernel::SpectrumOp::Eigen) &&
         RunSpectrum<rund::kernel::i64, 3u>(pick, rund::kernel::SpectrumOp::Eigen) &&
         RunSpectrum<rund::kernel::i64, 9u>(pick, rund::kernel::SpectrumOp::Eigen) &&
         RunSpectrum<rund::kernel::i64, 2u>(pick, rund::kernel::SpectrumOp::SVD) &&
         RunSpectrum<rund::kernel::i64, 3u>(pick, rund::kernel::SpectrumOp::SVD) &&
         RunSpectrum<rund::kernel::i64, 9u>(pick, rund::kernel::SpectrumOp::SVD) &&
         RunSpectrumVectors<rund::kernel::i64, 3u>(pick,
                                                   rund::kernel::SpectrumOp::Eigen) &&
         RunSpectrumVectors<rund::kernel::i64, 3u>(pick, rund::kernel::SpectrumOp::SVD);
}

[[nodiscard]] bool SpectrumRejectsInvalidShape(const rund::AccelDevice &pick) {
  namespace fix = node_accel_contract::primitive;
  auto context = rund::node::accel::OpenAccel(pick);
  if (!context.check.ok) {
    return false;
  }
  auto input = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::ReadOnly,
                               sizeof(rund::kernel::i32), 6u));
  auto values = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::WriteOnly,
                               sizeof(rund::kernel::i32), 2u));
  auto status = rund::node::accel::CreateAccelBuffer(
      context, fix::BufferDesc(rund::BufferUsage::WriteOnly,
                               sizeof(rund::kernel::u32), 1u));
  const std::array<rund::AccelGraphBufferRef, 3u> refs{
      rund::AccelRead(input, "matrix"), rund::AccelWrite(values, "values"),
      rund::AccelWrite(status, "status")};
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelSpectrum(
          refs.data(), refs.size(),
          rund::kernel::SpectrumDesc{.op = rund::kernel::SpectrumOp::Eigen,
                             .domain = rund::kernel::SpectrumDomain::SymmetricReal,
                             .vectors = rund::kernel::SpectrumVectors::ValuesOnly,
                             .rows = 2u,
                             .cols = 3u,
                             .max_iterations = 8u}),
  };
  const auto kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{.nodes = nodes.data(),
                                .node_count = nodes.size(),
                                .scalar = rund::kernel::ComputeScalar::Lane32,
                                .domain = rund::kernel::ComputeDomain::Fixed,
                                .fixed_format = test::FixedFormatForLane(
                                    rund::kernel::ComputeScalar::Lane32),
});
  return !kernel.check.ok;
}

[[nodiscard]] bool AvailableBackendsRunSpectrumNatively() {
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
    if (!BackendRunsSpectrum(pick)) {
      return false;
    }
  }
  return true;
}

} // namespace node_accel_contract
