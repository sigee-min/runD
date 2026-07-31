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

#define RUND_FUSION_CHECK(expr)                                                \
  do {                                                                         \
    if (!(expr)) {                                                             \
      return false;                                                            \
    }                                                                          \
  } while (false)

[[nodiscard]] bool RunExtraReadCase(
    const rund::AccelContext &context, const rund::compute_dsl::ComputeOp &op,
    const rund::compute_dsl::ComputeOp &two_read_op, const Inputs &inputs) {
  const rund::AccelBuffer input =
      MakeBuffer(context, rund::BufferUsage::ReadOnly);
  const rund::AccelBuffer mid =
      MakeBuffer(context, rund::BufferUsage::ReadWrite);
  const rund::AccelBuffer vel =
      MakeBuffer(context, rund::BufferUsage::ReadOnly);
  const rund::AccelBuffer output =
      MakeBuffer(context, rund::BufferUsage::WriteOnly);
  RUND_FUSION_CHECK(input.check.ok && mid.check.ok && vel.check.ok &&
                    output.check.ok);
  RUND_FUSION_CHECK(rund::node::accel::UploadAccelBuffer(
                        context, input, inputs.host.data(), sizeof(inputs.host))
                        .ok);
  RUND_FUSION_CHECK(rund::node::accel::UploadAccelBuffer(
                        context, vel, inputs.vel.data(), sizeof(inputs.vel))
                        .ok);

  std::array<GraphBufferRef, 5u> refs{
      GraphBufferRef{
          .buffer = &input, .role = Role::Read, .binding_name = "input"},
      GraphBufferRef{
          .buffer = &mid,
          .role = Role::Write,
          .binding_name = "output",
          .visibility = Visibility::Internal,
      },
      GraphBufferRef{
          .buffer = &mid,
          .role = Role::Read,
          .binding_name = "pos",
          .visibility = Visibility::Internal,
      },
      GraphBufferRef{.buffer = &vel, .role = Role::Read, .binding_name = "vel"},
      GraphBufferRef{
          .buffer = &output, .role = Role::Write, .binding_name = "output"},
  };
  std::array<GraphNode, 2u> nodes{
      rund::AccelMap(op.ir(), refs.data(), 2u, output.count),
      rund::AccelMap(two_read_op.ir(), refs.data() + 2u, 3u, output.count),
  };
  const rund::AccelKernel kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = nodes.data(),
                   .node_count = nodes.size(),
                   .scalar = op.ir().scalar,
                   .domain = op.ir().domain,
                   .fixed_format = op.ir().fixed_format,
               });
  RUND_FUSION_CHECK(kernel.check.ok);
  const rund::node::accel::detail::KernelExecution execution =
      rund::node::accel::detail::AdmitKernelForExecution(context, kernel);
  RUND_FUSION_CHECK(execution.admission.check.ok);
  RUND_FUSION_CHECK(execution.steps.size() == 1u);
  RUND_FUSION_CHECK(execution.removed_dispatch_count == 1u);
  const rund::node::accel::detail::KernelExecutionStep &step =
      execution.steps.front();
  const auto &accesses = step.artifact.metadata.binding_accesses;
  RUND_FUSION_CHECK(accesses.size() == 3u &&
                    step.graph_binding_indices.size() == 3u);
  RUND_FUSION_CHECK(accesses[0] == rund::kernel::ComputeBindingAccess::Read &&
                    accesses[1] == rund::kernel::ComputeBindingAccess::Read &&
                    accesses[2] == rund::kernel::ComputeBindingAccess::Write);
  RUND_FUSION_CHECK(step.graph_binding_indices[0] == 0u &&
                    step.graph_binding_indices[1] == 3u &&
                    step.graph_binding_indices[2] == 4u);

  std::array<KernelBinding, 5u> bindings{
      KernelBinding{.buffer = &input, .role = Role::Read},
      KernelBinding{.buffer = &mid, .role = Role::Write},
      KernelBinding{.buffer = &mid, .role = Role::Read},
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
  RUND_FUSION_CHECK(evidence.ok);
  RUND_FUSION_CHECK(evidence.original_operation_count == 2u);
  RUND_FUSION_CHECK(evidence.fused_operation_count == 1u);
  RUND_FUSION_CHECK(evidence.final_dispatch_count == 1u);
  std::array<rund::kernel::i32, 8u> download{};
  RUND_FUSION_CHECK(rund::node::accel::DownloadAccelBuffer(
                        context, output, download.data(), sizeof(download))
                        .ok);
  RUND_FUSION_CHECK(download == inputs.add7_vel);
  return true;
}

#undef RUND_FUSION_CHECK

} // namespace node_accel_contract::fusion
