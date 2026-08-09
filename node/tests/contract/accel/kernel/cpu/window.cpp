#include <accel/graph/factory/primitive/window.hpp>

#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/check.hpp>
#include <accel/context/buffer.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>

#include <node/accel/context.hpp>

#include "local.hpp"

#include <array>

namespace node_accel_contract::cpu_context {

bool CpuContextRunsWindow(const rund::AccelDevice &pick) {
  constexpr std::array<rund::kernel::u32, 6u> input{1u, 2u, 3u, 4u, 5u, 6u};
  constexpr std::array<rund::kernel::u32, 4u> expected{4u, 9u, 15u, 18u};
  rund::AccelContext context = rund::node::accel::OpenAccel(pick);
  rund::AccelBuffer source = rund::node::accel::CreateAccelBuffer(
      context, rund::AccelBufferDesc{
                   .scalar_width_bytes = sizeof(input[0u]),
                   .count = input.size(),
                   .usage = rund::BufferUsage::ReadOnly,
               });
  rund::AccelBuffer output = rund::node::accel::CreateAccelBuffer(
      context, rund::AccelBufferDesc{
                   .scalar_width_bytes = sizeof(expected[0u]),
                   .count = expected.size(),
                   .usage = rund::BufferUsage::WriteOnly,
               });
  if (!context.check.ok || !source.check.ok || !output.check.ok ||
      !rund::node::accel::UploadAccelBuffer(context, source, input.data(),
                                            input.size() * sizeof(input[0u]))
           .ok) {
    return false;
  }

  const std::array<rund::AccelGraphBufferRef, 2u> refs{
      rund::AccelGraphBufferRef{.buffer = &source,
                                .role = rund::kernel::BufferRole::Read},
      rund::AccelGraphBufferRef{.buffer = &output,
                                .role = rund::kernel::BufferRole::Write}};
  const rund::AccelGraphNode node =
      rund::AccelWindow(refs.data(), refs.size(),
                        rund::kernel::WindowDesc{
                            .op = rund::kernel::WindowOp::Sum,
                            .element = rund::kernel::WindowElement::U32,
                            .boundary = rund::kernel::WindowBoundary::Clamp,
                            .domain = rund::kernel::ComputeDomain::U32,
                            .input_count = input.size(),
                            .output_count = expected.size(),
                            .window_size = 3u,
                            .stride = 2u,
                            .pad_left = 1u,
                        });
  const rund::AccelKernel kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = &node,
                   .node_count = 1u,
                   .scalar = rund::kernel::ComputeScalar::Lane32,
                   .domain = rund::kernel::ComputeDomain::U32,
               });
  if (!kernel.check.ok) {
    return false;
  }

  const std::array<rund::AccelRunBinding, 2u> bindings{
      rund::AccelRunBinding{.buffer = &source,
                            .role = rund::kernel::BufferRole::Read},
      rund::AccelRunBinding{.buffer = &output,
                            .role = rund::kernel::BufferRole::Write}};
  const rund::AccelEvidence evidence =
      rund::node::accel::RunAccelKernel(context, kernel,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = expected.size(),
                                            .fresh_evidence = true,
                                        });
  if (!evidence.ok || evidence.backend != rund::AccelApi::Cpu ||
      evidence.command_submit_count != 0u || evidence.dispatch_count != 1u ||
      evidence.original_dispatch_count != 1u ||
      evidence.final_dispatch_count != 1u) {
    return false;
  }

  std::array<rund::kernel::u32, expected.size()> downloaded{};
  const rund::AccelCheck download = rund::node::accel::DownloadAccelBuffer(
      context, output, downloaded.data(),
      downloaded.size() * sizeof(downloaded[0u]));
  if (!download.ok || downloaded != expected) {
    return false;
  }

  rund::AccelGraphNode forged = node;
  forged.primitive_hash_lo ^= 1u;
  const rund::AccelKernel bad_hash = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = &forged,
                   .node_count = 1u,
                   .scalar = rund::kernel::ComputeScalar::Lane32,
                   .domain = rund::kernel::ComputeDomain::U32,
               });
  if (!KernelReason(bad_hash, "accel_kernel_graph_invalid")) {
    return false;
  }

  const rund::AccelGraphNode zero =
      rund::AccelWindow(nullptr, 0u,
                        rund::kernel::WindowDesc{
                            .op = rund::kernel::WindowOp::Sum,
                            .element = rund::kernel::WindowElement::U32,
                            .boundary = rund::kernel::WindowBoundary::Clamp,
                            .domain = rund::kernel::ComputeDomain::U32,
                            .input_count = 0u,
                            .output_count = 0u,
                            .window_size = 1u,
                            .stride = 1u,
                            .pad_left = 0u,
                        });
  const rund::AccelKernel zero_graph = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = &zero,
                   .node_count = 1u,
                   .scalar = rund::kernel::ComputeScalar::Lane32,
                   .domain = rund::kernel::ComputeDomain::U32,
               });
  return KernelReason(zero_graph, "accel_kernel_graph_invalid");
}

} // namespace node_accel_contract::cpu_context
