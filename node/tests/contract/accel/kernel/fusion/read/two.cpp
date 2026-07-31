#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/value.hpp>

#include <accel/graph/factory/map.hpp>

#include "../local.hpp"
#include <node/accel/context.hpp>

#include <array>

namespace node_accel_contract::fusion {

[[nodiscard]] bool
RunTwoReadCase(const rund::AccelContext &context,
               const rund::compute_dsl::ComputeOp &two_read_op,
               const Inputs &inputs) {
  const rund::AccelBuffer pos =
      MakeBuffer(context, rund::BufferUsage::ReadOnly);
  const rund::AccelBuffer vel =
      MakeBuffer(context, rund::BufferUsage::ReadOnly);
  const rund::AccelBuffer output =
      MakeBuffer(context, rund::BufferUsage::WriteOnly);
  if (!pos.check.ok || !vel.check.ok || !output.check.ok ||
      !rund::node::accel::UploadAccelBuffer(context, pos, inputs.host.data(),
                                            sizeof(inputs.host))
           .ok ||
      !rund::node::accel::UploadAccelBuffer(context, vel, inputs.vel.data(),
                                            sizeof(inputs.vel))
           .ok) {
    return false;
  }
  std::array<GraphBufferRef, 3u> refs{
      GraphBufferRef{.buffer = &pos, .role = Role::Read, .binding_name = "pos"},
      GraphBufferRef{.buffer = &vel, .role = Role::Read, .binding_name = "vel"},
      GraphBufferRef{
          .buffer = &output, .role = Role::Write, .binding_name = "output"},
  };
  std::array<GraphNode, 1u> nodes{
      rund::AccelMap(two_read_op.ir(), refs.data(), refs.size(), output.count),
  };
  const rund::AccelKernel kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = nodes.data(),
                   .node_count = nodes.size(),
                   .scalar = two_read_op.ir().scalar,
                   .domain = two_read_op.ir().domain,
                   .fixed_format = two_read_op.ir().fixed_format,
               });
  if (!kernel.check.ok) {
    return false;
  }
  std::array<KernelBinding, 3u> bindings{
      KernelBinding{.buffer = &pos, .role = Role::Read},
      KernelBinding{.buffer = &vel, .role = Role::Read},
      KernelBinding{.buffer = &output, .role = Role::Write},
  };
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(context, kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = inputs.host.size(),
                                            .fresh_evidence = true,
                                        });
  std::array<rund::kernel::i32, 8u> download{};
  return evidence.ok && evidence.dispatch_count == 1u &&
         evidence.device_to_host_bytes == 0u &&
         rund::node::accel::DownloadAccelBuffer(
             context, output, download.data(), sizeof(download))
             .ok &&
         download == inputs.add_vel;
}

} // namespace node_accel_contract::fusion
