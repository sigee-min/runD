#include <accel/api.hpp>
#include <accel/device.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/value.hpp>

#include <accel/graph/factory/primitive/factor/node.hpp>
#include <accel/graph/factory/primitive/solve/node.hpp>
#include <accel/graph/factory/primitive/spectrum/node.hpp>

#include "test/compute/fixed.hpp"
#include <node/accel/context.hpp>

#include <array>
#include <string_view>

namespace node_accel_contract {

[[nodiscard]] bool
NumericAlgebraRejectsOversizeShape(const rund::AccelDevice &pick) {
  auto context = rund::node::accel::OpenAccel(pick);
  if (!context.check.ok) {
    return false;
  }
  const std::array<rund::AccelGraphNode, 1u> factor{
      rund::AccelFactor(
          nullptr, 0u,
          rund::kernel::FactorDesc{.op = rund::kernel::FactorOp::QR,
                                   .pivot = rund::kernel::PivotOp::None,
                                   .rows = 17u,
                                   .cols = 17u}),
  };
  const std::array<rund::AccelGraphNode, 1u> solve{
      rund::AccelSolve(
          nullptr, 0u,
          rund::kernel::SolveDesc{.op = rund::kernel::SolveOp::Linear,
                                  .input = rund::kernel::SolveInput::Matrix,
                                  .factor = rund::kernel::FactorOp::LU,
                                  .rows = 17u,
                                  .rhs_cols = 1u}),
  };
  const std::array<rund::AccelGraphNode, 1u> spectrum{
      rund::AccelSpectrum(
          nullptr, 0u,
          rund::kernel::SpectrumDesc{
              .op = rund::kernel::SpectrumOp::Eigen,
              .domain = rund::kernel::SpectrumDomain::SymmetricReal,
              .vectors = rund::kernel::SpectrumVectors::ValuesOnly,
              .rows = 17u,
              .cols = 17u,
              .max_iterations = 32u}),
  };
  const auto rejects = [&](const auto &nodes) {
    const auto kernel = rund::node::accel::CompileAccelKernel(
        context, rund::AccelGraph{
                     .nodes = nodes.data(),
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

} // namespace node_accel_contract
