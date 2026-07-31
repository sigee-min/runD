#pragma once

#include <accel/graph/factory/map.hpp>

#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/value.hpp>

#include <node/accel/context.hpp>

#include "../local.hpp"

#include <array>
#include <string_view>

namespace node_accel_contract::fusion {

[[nodiscard]] bool RunConflictCase(const rund::AccelContext &context,
                                   const rund::compute_dsl::ComputeOp &op,
                                   const Inputs &inputs) {
  const rund::AccelBuffer input =
      MakeBuffer(context, rund::BufferUsage::ReadOnly);
  const rund::AccelBuffer mid =
      MakeBuffer(context, rund::BufferUsage::ReadWrite);
  const rund::AccelBuffer output =
      MakeBuffer(context, rund::BufferUsage::WriteOnly);
  const rund::AccelBuffer branch =
      MakeBuffer(context, rund::BufferUsage::WriteOnly);
  if (!input.check.ok || !mid.check.ok || !output.check.ok ||
      !branch.check.ok) {
    return false;
  }
  if (!rund::node::accel::UploadAccelBuffer(context, input, inputs.host.data(),
                                            sizeof(inputs.host))
           .ok) {
    return false;
  }
  std::array<GraphBufferRef, 6u> refs{
      GraphBufferRef{.buffer = &input, .role = Role::Read},
      GraphBufferRef{.buffer = &mid,
                     .role = Role::Write,
                     .visibility = Visibility::Internal},
      GraphBufferRef{.buffer = &mid,
                     .role = Role::Read,
                     .visibility = Visibility::Internal},
      GraphBufferRef{.buffer = &output, .role = Role::Write},
      GraphBufferRef{.buffer = &mid,
                     .role = Role::Read,
                     .visibility = Visibility::Internal},
      GraphBufferRef{.buffer = &branch, .role = Role::Write},
  };
  std::array<GraphNode, 3u> nodes{
      rund::AccelMap(op.ir(), refs.data(), 2u, output.count),
      rund::AccelMap(op.ir(), refs.data() + 2u, 2u, output.count),
      rund::AccelMap(op.ir(), refs.data() + 4u, 2u, output.count),
  };
  const rund::AccelKernel kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = nodes.data(),
                   .node_count = nodes.size(),
                   .scalar = op.ir().scalar,
                   .domain = op.ir().domain,
                   .fixed_format = op.ir().fixed_format,
               });
  if (!kernel.check.ok || kernel.node_count != 3u) {
    return false;
  }
  std::array<KernelBinding, 6u> bindings{
      KernelBinding{.buffer = &input, .role = Role::Read},
      KernelBinding{.buffer = &mid, .role = Role::Write},
      KernelBinding{.buffer = &mid, .role = Role::Read},
      KernelBinding{.buffer = &output, .role = Role::Write},
      KernelBinding{.buffer = &mid, .role = Role::Read},
      KernelBinding{.buffer = &branch, .role = Role::Write},
  };
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(context, kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = inputs.host.size(),
                                            .fresh_evidence = true,
                                        });
  if (!evidence.ok || evidence.original_operation_count != 3u ||
      evidence.fused_operation_count != 3u ||
      evidence.original_dispatch_count != 3u ||
      evidence.final_dispatch_count != 3u || evidence.dispatch_count != 3u ||
      evidence.internal_producer_consumer_roundtrip_bytes !=
          inputs.host.size() * sizeof(rund::kernel::i32) * 4u ||
      evidence.external_producer_consumer_roundtrip_bytes != 0u ||
      evidence.fusion_rejection_count != 1u ||
      std::string_view{evidence.fusion_reason} !=
          "compute_fusion_dependency_conflict") {
    return false;
  }
  std::array<rund::kernel::i32, 8u> output_download{};
  std::array<rund::kernel::i32, 8u> branch_download{};
  return rund::node::accel::DownloadAccelBuffer(
             context, output, output_download.data(), sizeof(output_download))
             .ok &&
         rund::node::accel::DownloadAccelBuffer(
             context, branch, branch_download.data(), sizeof(branch_download))
             .ok &&
         output_download == inputs.add14 && branch_download == inputs.add14;
}

} // namespace node_accel_contract::fusion
