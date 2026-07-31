#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/value.hpp>

#include <accel/graph/factory/map.hpp>

#include "local.hpp"
#include <node/accel/context.hpp>

#include <array>
#include <cstddef>
#include <string_view>

namespace node_accel_contract::fusion {

[[nodiscard]] bool RunVisibilityCase(const rund::AccelContext &context,
                                     const bool internal_intermediate) {
  const rund::compute_dsl::ComputeOp first = BuildAddOp(
      internal_intermediate ? "fusion-internal-a" : "fusion-external-a");
  const rund::compute_dsl::ComputeOp second = BuildAddOp(
      internal_intermediate ? "fusion-internal-b" : "fusion-external-b");
  if (!first.ok() || !second.ok()) {
    return false;
  }

  const rund::AccelBuffer input =
      rund::node::accel::CreateAccelBuffer(context, BufferDesc());
  const rund::AccelBuffer mid =
      rund::node::accel::CreateAccelBuffer(context, BufferDesc());
  const rund::AccelBuffer output =
      rund::node::accel::CreateAccelBuffer(context, BufferDesc());
  if (!input.check.ok || !mid.check.ok || !output.check.ok) {
    return false;
  }

  std::array<rund::kernel::i32, 8u> host_input{};
  std::array<rund::kernel::i32, 8u> expected_mid{};
  std::array<rund::kernel::i32, 8u> expected_output{};
  for (std::size_t index = 0u; index < host_input.size(); ++index) {
    host_input[index] = static_cast<rund::kernel::i32>(index * 3u + 1u);
    expected_mid[index] = host_input[index] + 7;
    expected_output[index] = host_input[index] + 14;
  }
  if (!rund::node::accel::UploadAccelBuffer(context, input, host_input.data(),
                                            sizeof(host_input))
           .ok) {
    return false;
  }

  const Visibility mid_visibility =
      internal_intermediate ? Visibility::Internal : Visibility::External;
  std::array<GraphBufferRef, 4u> refs{
      GraphBufferRef{.buffer = &input, .role = Role::Read},
      GraphBufferRef{
          .buffer = &mid, .role = Role::Write, .visibility = mid_visibility},
      GraphBufferRef{
          .buffer = &mid, .role = Role::Read, .visibility = mid_visibility},
      GraphBufferRef{.buffer = &output, .role = Role::Write},
  };
  std::array<GraphNode, 2u> nodes{
      rund::AccelMap(first.ir(), refs.data(), 2u, output.count),
      rund::AccelMap(second.ir(), refs.data() + 2u, 2u, output.count),
  };
  const rund::AccelKernel kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = nodes.data(),
                   .node_count = nodes.size(),
                   .scalar = first.ir().scalar,
                   .domain = first.ir().domain,
                   .fixed_format = first.ir().fixed_format,
               });
  if (!kernel.check.ok) {
    return false;
  }
  std::array<KernelBinding, 4u> bindings{
      KernelBinding{.buffer = &input, .role = Role::Read},
      KernelBinding{.buffer = &mid, .role = Role::Write},
      KernelBinding{.buffer = &mid, .role = Role::Read},
      KernelBinding{.buffer = &output, .role = Role::Write},
  };
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(context, kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = host_input.size(),
                                            .fresh_evidence = true,
                                        });
  if (!evidence.ok) {
    return false;
  }
  if (internal_intermediate) {
    if (evidence.original_operation_count != 2u ||
        evidence.fused_operation_count != 1u ||
        evidence.final_dispatch_count != 1u ||
        evidence.dispatch_count != 1u ||
        std::string_view{evidence.fusion_reason} != "compute_fusion_ok") {
      return false;
    }
  } else {
    if (evidence.original_operation_count != 2u ||
        evidence.fused_operation_count != 2u ||
        evidence.final_dispatch_count != 2u ||
        evidence.dispatch_count != 2u ||
        evidence.fusion_rejection_count != 1u ||
        std::string_view{evidence.fusion_reason} !=
            "compute_fusion_visibility_boundary") {
      return false;
    }
    std::array<rund::kernel::i32, 8u> mid_download{};
    if (!rund::node::accel::DownloadAccelBuffer(
             context, mid, mid_download.data(), sizeof(mid_download))
             .ok ||
        mid_download != expected_mid) {
      return false;
    }
  }

  std::array<rund::kernel::i32, 8u> output_download{};
  return rund::node::accel::DownloadAccelBuffer(
             context, output, output_download.data(), sizeof(output_download))
             .ok &&
         output_download == expected_output;
}

[[nodiscard]] bool RunRegionCase(const rund::AccelContext &context) {
  const rund::compute_dsl::ComputeOp op = BuildAddOp("fusion-region");
  if (!op.ok()) {
    return false;
  }

  std::array<rund::AccelBuffer, 5u> buffers{};
  for (rund::AccelBuffer &buffer : buffers) {
    buffer = rund::node::accel::CreateAccelBuffer(context, BufferDesc());
    if (!buffer.check.ok) {
      return false;
    }
  }

  std::array<rund::kernel::i32, 8u> input{};
  std::array<rund::kernel::i32, 8u> expected_boundary{};
  std::array<rund::kernel::i32, 8u> expected_output{};
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<rund::kernel::i32>(index * 5u + 1u);
    expected_boundary[index] = input[index] + 14;
    expected_output[index] = input[index] + 28;
  }
  if (!rund::node::accel::UploadAccelBuffer(context, buffers[0], input.data(),
                                            sizeof(input))
           .ok) {
    return false;
  }

  std::array<GraphBufferRef, 8u> refs{
      GraphBufferRef{.buffer = &buffers[0], .role = Role::Read},
      GraphBufferRef{.buffer = &buffers[1],
                     .role = Role::Write,
                     .visibility = Visibility::Internal},
      GraphBufferRef{.buffer = &buffers[1],
                     .role = Role::Read,
                     .visibility = Visibility::Internal},
      GraphBufferRef{.buffer = &buffers[2],
                     .role = Role::Write,
                     .visibility = Visibility::External},
      GraphBufferRef{.buffer = &buffers[2],
                     .role = Role::Read,
                     .visibility = Visibility::External},
      GraphBufferRef{.buffer = &buffers[3],
                     .role = Role::Write,
                     .visibility = Visibility::Internal},
      GraphBufferRef{.buffer = &buffers[3],
                     .role = Role::Read,
                     .visibility = Visibility::Internal},
      GraphBufferRef{.buffer = &buffers[4], .role = Role::Write},
  };
  std::array<GraphNode, 4u> nodes{
      rund::AccelMap(op.ir(), refs.data(), 2u, input.size()),
      rund::AccelMap(op.ir(), refs.data() + 2u, 2u, input.size()),
      rund::AccelMap(op.ir(), refs.data() + 4u, 2u, input.size()),
      rund::AccelMap(op.ir(), refs.data() + 6u, 2u, input.size()),
  };
  const rund::AccelKernel kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = nodes.data(),
                   .node_count = nodes.size(),
                   .scalar = op.ir().scalar,
                   .domain = op.ir().domain,
                   .fixed_format = op.ir().fixed_format,
               });
  if (!kernel.check.ok) {
    return false;
  }
  const rund::node::accel::detail::KernelExecution execution =
      rund::node::accel::detail::AdmitKernelForExecution(context, kernel);
  if (!execution.admission.check.ok || execution.steps.size() != 2u ||
      execution.removed_dispatch_count != 2u ||
      execution.original_operation_count != 4u ||
      execution.fused_operation_count != 2u ||
      execution.fusion_rejection_count != 1u ||
      std::string_view{execution.fusion_reason} !=
          "compute_fusion_visibility_boundary") {
    return false;
  }

  std::array<KernelBinding, 8u> bindings{};
  for (std::size_t index = 0u; index < bindings.size(); ++index) {
    bindings[index] = KernelBinding{
        .buffer = &buffers[(index + 1u) / 2u],
        .role = index % 2u == 0u ? Role::Read : Role::Write,
    };
  }
  const rund::AccelEvidence evidence = rund::node::accel::RunAccelKernel(
      context, kernel,
      rund::AccelRun{
          .bindings = bindings.data(),
          .binding_count = bindings.size(),
          .tile_count = input.size(),
          .fresh_evidence = true,
      });
  if (!evidence.ok || evidence.original_operation_count != 4u ||
      evidence.fused_operation_count != 2u ||
      evidence.original_dispatch_count != 4u ||
      evidence.final_dispatch_count != 2u || evidence.dispatch_count != 2u ||
      evidence.fusion_rejection_count != 1u) {
    return false;
  }

  std::array<rund::kernel::i32, 8u> boundary{};
  std::array<rund::kernel::i32, 8u> output{};
  return rund::node::accel::DownloadAccelBuffer(
             context, buffers[2], boundary.data(), sizeof(boundary))
             .ok &&
         rund::node::accel::DownloadAccelBuffer(
             context, buffers[4], output.data(), sizeof(output))
             .ok &&
         boundary == expected_boundary && output == expected_output;
}

} // namespace node_accel_contract::fusion
