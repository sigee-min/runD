#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/visibility.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/run/binding.hpp>
#include <accel/kernel/value.hpp>

#include <accel/graph/factory/buffer/read.hpp>
#include <accel/graph/factory/buffer/write.hpp>
#include <accel/graph/factory/map.hpp>

#include "local.hpp"
#include <node/accel/context.hpp>

#include <array>
#include <memory>

namespace node_accel_contract::kernel_case::compile {

bool TamperedSupportRejects(const Fixture &fixture) {
  rund::AccelKernel tampered = fixture.first;
  ++tampered.graph_id_hi;
  if (!CheckReason(rund::node::accel::detail::AdmitKernelForSupport(
                       fixture.context, tampered)
                       .check,
                   "accel_kernel_graph_invalid")) {
    return false;
  }
  tampered = fixture.first;
  ++tampered.frozen_caps.max_window_tiles;
  if (!CheckReason(rund::node::accel::detail::AdmitKernelForSupport(
                       fixture.context, tampered)
                       .check,
                   "accel_kernel_graph_invalid")) {
    return false;
  }
  tampered = fixture.first;
  tampered.owner =
      std::shared_ptr<void>(fixture.first.owner.get(), [](void *) {});
  return CheckReason(rund::node::accel::detail::AdmitKernelForSupport(
                         fixture.context, tampered)
                         .check,
                     "accel_kernel_graph_invalid");
}

bool ForeignBufferRejects(const Fixture &fixture) {
  const rund::AccelContext second = rund::node::accel::OpenAccel(fixture.pick);
  if (!second.check.ok) {
    return false;
  }
  if (!CheckReason(rund::node::accel::detail::AdmitKernelForSupport(
                       second, fixture.first)
                       .check,
                   "accel_kernel_graph_invalid")) {
    return false;
  }

  const rund::AccelBuffer foreign =
      rund::node::accel::CreateAccelBuffer(second, BufferDesc());
  if (!foreign.check.ok) {
    return false;
  }
  std::array<rund::AccelRunBinding, 2u> bindings{
      rund::AccelRunBinding{
          .buffer = &foreign,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelRunBinding{
          .buffer = &fixture.output,
          .role = rund::kernel::BufferRole::Write,
      },
  };
  const rund::AccelEvidence run =
      rund::node::accel::RunAccelKernel(fixture.context, fixture.first,
                                        rund::AccelRun{
                                            .bindings = bindings.data(),
                                            .binding_count = bindings.size(),
                                            .tile_count = fixture.input.count,
                                            .fresh_evidence = true,
                                        });
  if (!EvidenceReason(run, "accel_kernel_buffer_owner_mismatch")) {
    return false;
  }

  std::array<rund::AccelGraphBufferRef, 2u> refs{};
  std::array<rund::AccelGraphNode, 1u> nodes{};
  const rund::AccelGraph graph =
      GraphFor(fixture.op.ir(), foreign, fixture.output, refs, nodes);
  return CheckReason(
      rund::node::accel::CompileAccelKernel(fixture.context, graph).check,
      "accel_kernel_buffer_owner_mismatch");
}

bool LogicalBufferIdentityRejects(const Fixture &fixture) {
  const auto compile = [&](const auto &refs) {
    std::array<rund::AccelGraphNode, 1u> nodes{rund::AccelMap(
        fixture.op.ir(), refs.data(), refs.size(), fixture.input.count)};
    return rund::node::accel::CompileAccelKernel(
        fixture.context, rund::AccelGraph{
                             .nodes = nodes.data(),
                             .node_count = nodes.size(),
                             .scalar = fixture.op.ir().scalar,
                             .domain = fixture.op.ir().domain,
                             .fixed_format = fixture.op.ir().fixed_format,
                         });
  };

  std::array<rund::AccelGraphBufferRef, 2u> shape_mismatch{
      rund::AccelRead(BufferDesc(rund::BufferUsage::ReadOnly), "input",
                      rund::GraphBufferVisibility::External, 17u),
      rund::AccelWrite(
          BufferDesc(rund::BufferUsage::WriteOnly, fixture.output.count - 1u),
          "output", rund::GraphBufferVisibility::External, 17u),
  };
  if (!CheckReason(compile(shape_mismatch).check,
                   "accel_kernel_graph_invalid")) {
    return false;
  }

  std::array<rund::AccelGraphBufferRef, 2u> visibility_mismatch{
      rund::AccelRead(fixture.input, "input",
                      rund::GraphBufferVisibility::External, 19u),
      rund::AccelWrite(fixture.output, "output",
                       rund::GraphBufferVisibility::Internal, 19u),
  };
  if (!CheckReason(compile(visibility_mismatch).check,
                   "accel_kernel_graph_invalid")) {
    return false;
  }

  std::array<rund::AccelGraphBufferRef, 2u> read_reset = fixture.refs;
  read_reset[0u].init = rund::kernel::BufferInit::Zero;
  if (!CheckReason(compile(read_reset).check, "accel_kernel_graph_invalid")) {
    return false;
  }

  std::array<rund::AccelGraphBufferRef, 2u> shape_only_internal{
      rund::AccelRead(fixture.input, "input"),
      rund::AccelWrite(BufferDesc(rund::BufferUsage::WriteOnly), "output",
                       rund::GraphBufferVisibility::Internal, 23u),
  };
  if (!compile(shape_only_internal).check.ok) {
    return false;
  }

  std::array<rund::AccelGraphBufferRef, 4u> duplicate{
      fixture.refs[0u],
      fixture.refs[1u],
      fixture.refs[0u],
      fixture.refs[1u],
  };
  duplicate[1u].logical_id = 29u;
  duplicate[1u].init = rund::kernel::BufferInit::Zero;
  duplicate[3u].logical_id = 29u;
  duplicate[3u].init = rund::kernel::BufferInit::Zero;
  std::array<rund::AccelGraphNode, 2u> nodes{
      rund::AccelMap(fixture.op.ir(), duplicate.data(), 2u,
                     fixture.input.count),
      rund::AccelMap(fixture.op.ir(), duplicate.data() + 2u, 2u,
                     fixture.input.count),
  };
  const rund::AccelKernel duplicate_reset =
      rund::node::accel::CompileAccelKernel(
          fixture.context, rund::AccelGraph{
                               .nodes = nodes.data(),
                               .node_count = nodes.size(),
                               .scalar = fixture.graph.scalar,
                               .domain = fixture.graph.domain,
                               .fixed_format = fixture.graph.fixed_format,
                           });
  return CheckReason(duplicate_reset.check, "accel_kernel_graph_invalid");
}

bool UnsupportedAndInvalidGraphRejects(const Fixture &fixture) {
  std::array<rund::AccelGraphBufferRef, 2u> refs{};
  std::array<rund::AccelGraphNode, 1u> nodes{};
  const rund::kernel::ComputeIR unsupported = UnsupportedIr(fixture.op.ir());
  const rund::AccelGraph unsupported_graph =
      GraphFor(unsupported, fixture.input, fixture.output, refs, nodes);
  if (!CheckReason(rund::node::accel::CompileAccelKernel(fixture.context,
                                                         unsupported_graph)
                       .check,
                   "accel_kernel_artifact_invalid")) {
    return false;
  }

  std::array<rund::AccelGraphNode, 1u> mixed_nodes = fixture.nodes;
  mixed_nodes[0].segmented_reduce = rund::kernel::SegmentedReduceDesc{
      .op = rund::kernel::ReduceOp::Max,
      .element = rund::kernel::ReduceElement::U32,
      .element_count = fixture.input.count,
      .block_size = fixture.input.count,
  };
  const rund::AccelGraph mixed_graph{
      .nodes = mixed_nodes.data(),
      .node_count = mixed_nodes.size(),
      .scalar = fixture.graph.scalar,
      .domain = fixture.graph.domain,
      .fixed_format = fixture.graph.fixed_format,
  };
  if (!CheckReason(
          rund::node::accel::CompileAccelKernel(fixture.context, mixed_graph)
              .check,
          "accel_kernel_graph_invalid")) {
    return false;
  }

  std::array<rund::AccelGraphNode, 1u> bad_signature_nodes = fixture.nodes;
  bad_signature_nodes[0].signature.values[0u].count += 1u;
  const rund::AccelGraph bad_signature_graph{
      .nodes = bad_signature_nodes.data(),
      .node_count = bad_signature_nodes.size(),
      .scalar = fixture.graph.scalar,
      .domain = fixture.graph.domain,
      .fixed_format = fixture.graph.fixed_format,
  };
  if (!CheckReason(rund::node::accel::CompileAccelKernel(fixture.context,
                                                         bad_signature_graph)
                       .check,
                   "accel_kernel_graph_invalid")) {
    return false;
  }

  std::array<rund::AccelGraphNode, 1u> missing_signature_nodes = fixture.nodes;
  missing_signature_nodes[0].signature = rund::kernel::GraphSignature{};
  const rund::AccelGraph missing_signature_graph{
      .nodes = missing_signature_nodes.data(),
      .node_count = missing_signature_nodes.size(),
      .scalar = fixture.graph.scalar,
      .domain = fixture.graph.domain,
      .fixed_format = fixture.graph.fixed_format,
  };
  if (!CheckReason(rund::node::accel::CompileAccelKernel(
                       fixture.context, missing_signature_graph)
                       .check,
                   "accel_kernel_graph_invalid")) {
    return false;
  }

  rund::AccelGraph invalid_graph = fixture.graph;
  invalid_graph.node_count = 0u;
  return CheckReason(
      rund::node::accel::CompileAccelKernel(fixture.context, invalid_graph)
          .check,
      "accel_kernel_graph_invalid");
}

} // namespace node_accel_contract::kernel_case::compile
