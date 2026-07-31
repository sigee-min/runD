#include <accel/context/buffer.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>
#include <accel/kernel/value.hpp>

#include <accel/graph/factory/map.hpp>
#include <kernel/program/compute/graph/validation.hpp>

#include "local.hpp"
#include <node/accel/context.hpp>

#include <array>

namespace node_accel_contract::kernel_case::compile {

bool MultiNodeGraphIdMatchesKernel(const Fixture &fixture) {
  const rund::AccelBuffer intermediate =
      rund::node::accel::CreateAccelBuffer(fixture.context, BufferDesc());
  if (!intermediate.check.ok) {
    return false;
  }

  std::array<rund::AccelGraphBufferRef, 4u> refs{
      rund::AccelGraphBufferRef{
          .buffer = &fixture.input,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelGraphBufferRef{
          .buffer = &intermediate,
          .role = rund::kernel::BufferRole::Write,
      },
      rund::AccelGraphBufferRef{
          .buffer = &intermediate,
          .role = rund::kernel::BufferRole::Read,
      },
      rund::AccelGraphBufferRef{
          .buffer = &fixture.output,
          .role = rund::kernel::BufferRole::Write,
      },
  };
  std::array<rund::AccelGraphNode, 2u> nodes{
      rund::AccelMap(fixture.op.ir(), refs.data(), 2u, intermediate.count),
      rund::AccelMap(fixture.op.ir(), refs.data() + 2u, 2u,
                     fixture.output.count),
  };
  const rund::AccelGraph graph{
      .nodes = nodes.data(),
      .node_count = nodes.size(),
      .scalar = fixture.op.ir().scalar,
      .domain = fixture.op.ir().domain,
      .fixed_format = fixture.op.ir().fixed_format,
  };
  const rund::AccelKernel multi =
      rund::node::accel::CompileAccelKernel(fixture.context, graph);
  if (!multi.check.ok || multi.node_count != 2u ||
      (multi.graph_id_hi == fixture.first.graph_id_hi &&
       multi.graph_id_lo == fixture.first.graph_id_lo)) {
    return false;
  }

  const std::array<rund::kernel::GraphBufferRef, 4u> expected_refs{
      rund::kernel::GraphBufferRef{.logical_id = 1u,
                                   .role = rund::kernel::BufferRole::Read},
      rund::kernel::GraphBufferRef{.logical_id = 2u,
                                   .role = rund::kernel::BufferRole::Write},
      rund::kernel::GraphBufferRef{.logical_id = 2u,
                                   .role = rund::kernel::BufferRole::Read},
      rund::kernel::GraphBufferRef{.logical_id = 3u,
                                   .role = rund::kernel::BufferRole::Write}};
  const std::array<rund::kernel::GraphNode, 2u> expected_nodes{
      rund::kernel::GraphNode{.op_hash_hi = fixture.op.ir().op_hash_hi,
                              .op_hash_lo = fixture.op.ir().op_hash_lo,
                              .buffers = expected_refs.data(),
                              .buffer_count = 2u,
                              .element_count = intermediate.count},
      rund::kernel::GraphNode{.op_hash_hi = fixture.op.ir().op_hash_hi,
                              .op_hash_lo = fixture.op.ir().op_hash_lo,
                              .buffers = expected_refs.data() + 2u,
                              .buffer_count = 2u,
                              .element_count = fixture.output.count}};
  const rund::kernel::GraphCheck expected = rund::kernel::ValidateGraph(
      rund::kernel::Graph{.nodes = expected_nodes.data(),
                          .node_count = expected_nodes.size(),
                          .scalar = fixture.op.ir().scalar,
                          .domain = fixture.op.ir().domain,
                          .fixed_format = fixture.op.ir().fixed_format,
});
  return expected.ok && multi.graph_id_hi == expected.graph_id_hi &&
         multi.graph_id_lo == expected.graph_id_lo;
}

} // namespace node_accel_contract::kernel_case::compile
