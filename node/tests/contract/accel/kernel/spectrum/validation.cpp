#include "local.hpp"

#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>

#include <accel/graph/factory/buffer/read.hpp>
#include <accel/graph/factory/buffer/write.hpp>
#include <accel/graph/factory/primitive/spectrum/node.hpp>

#include "../primitive/local.hpp"
#include "test/compute/fixed.hpp"
#include <node/accel/context.hpp>
#include <node/accel/pick.hpp>

#include <array>

namespace node_accel_contract {

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
          rund::kernel::SpectrumDesc{
              .op = rund::kernel::SpectrumOp::Eigen,
              .domain = rund::kernel::SpectrumDomain::SymmetricReal,
              .vectors = rund::kernel::SpectrumVectors::ValuesOnly,
              .rows = 2u,
              .cols = 3u,
              .max_iterations = 8u}),
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
