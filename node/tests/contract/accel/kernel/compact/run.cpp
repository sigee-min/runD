#include <accel/buffer.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>

#include <accel/graph/factory/primitive/compact.hpp>
#include <kernel/program/compute/compact/plan.hpp>

#include "local.hpp"
#include "test/compute/fixed.hpp"
#include <node/accel/context.hpp>

namespace node_accel_contract::compact {

RunContext Compile(const rund::AccelDevice &pick,
                   const rund::kernel::ComputeScalar scalar,
                   const std::array<rund::kernel::u32, 8u> &flags,
                   const std::uint64_t capacity) {
  namespace p = node_accel_contract::primitive;
  RunContext ctx{};
  ctx.context = rund::node::accel::OpenAccel(pick);
  if (!ctx.context.check.ok) {
    return ctx;
  }
  ctx.flags = rund::node::accel::CreateAccelBuffer(
      ctx.context, p::BufferDesc(rund::BufferUsage::ReadOnly,
                                 sizeof(rund::kernel::u32), flags.size()));
  ctx.output = rund::node::accel::CreateAccelBuffer(
      ctx.context, p::BufferDesc(rund::BufferUsage::WriteOnly,
                                 sizeof(rund::kernel::u32), capacity));
  if (!ctx.flags.check.ok || !ctx.output.check.ok ||
      !rund::node::accel::UploadAccelBuffer(
           ctx.context, ctx.flags, flags.data(),
           flags.size() * sizeof(rund::kernel::u32))
           .ok) {
    return ctx;
  }
  std::array<rund::AccelGraphBufferRef, 2u> refs{
      rund::AccelGraphBufferRef{
          .buffer = &ctx.flags,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelGraphBufferRef{
          .buffer = &ctx.output,
          .role = rund::kernel::BufferRole::Write,
      },
  };
  const rund::kernel::CompactDesc desc{
      .element_count = flags.size(),
      .output_capacity = capacity,
      .flag_bytes = sizeof(rund::kernel::u32),
      .output_bytes = sizeof(rund::kernel::u32),
  };
  ctx.plan = rund::kernel::PlanCompact(desc);
  const std::array<rund::AccelGraphNode, 1u> nodes{
      rund::AccelCompact(refs.data(), refs.size(), desc)};
  ctx.kernel = rund::node::accel::CompileAccelKernel(
      ctx.context, rund::AccelGraph{
                       .nodes = nodes.data(),
                       .node_count = nodes.size(),
                       .scalar = scalar,
                       .domain = rund::kernel::ComputeDomain::Fixed,
                       .fixed_format = test::FixedFormatForLane(scalar),
                   });
  ctx.valid = ctx.plan.ok && ctx.kernel.check.ok;
  return ctx;
}

rund::AccelEvidence Run(const RunContext &ctx, const std::uint64_t tile_count) {
  const std::array<rund::AccelRunBinding, 2u> bindings{
      rund::AccelRunBinding{
          .buffer = &ctx.flags,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelRunBinding{
          .buffer = &ctx.output,
          .role = rund::kernel::BufferRole::Write,
      },
  };
  return rund::node::accel::RunAccelKernel(ctx.context, ctx.kernel,
                                           rund::AccelRun{
                                               .bindings = bindings.data(),
                                               .binding_count = bindings.size(),
                                               .tile_count = tile_count,
                                               .fresh_evidence = true,
                                           });
}

} // namespace node_accel_contract::compact
