#pragma once

#include <accel/context/buffer/descriptor.hpp>
#include <accel/context/value.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/visibility.hpp>

#include <node/accel/context.hpp>

#include "../context/internal.hpp"
#include "operation.hpp"
#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/fusion.hpp>
#include <kernel/program/compute/graph/schema.hpp>
#include <kernel/program/compute/ir.hpp>
#include <kernel/program/compute/lowering/admission.hpp>
#include <kernel/program/compute/model.hpp>

#include <cstdint>
#include <map>
#include <set>
#include <vector>

namespace rund::node::accel::detail {

struct GraphCompileNode {
  Operation operation{};
  const rund::kernel::ComputeIR *ir = nullptr;
  rund::kernel::LoweringArtifact artifact{};
  rund::kernel::compute_lowering_detail::ComputeInputAdmission cpu_input{};
  rund::kernel::ExecutionMetadata map_metadata{};
  KernelBindingIndices binding_indices{};
  std::uint64_t primitive_hash_hi = 0u;
  std::uint64_t primitive_hash_lo = 0u;
  std::uint64_t element_count = 0u;
  rund::kernel::GraphControl control{};
  rund::kernel::GraphSignature signature{};
  bool fusion_supported = false;
  bool fusion_write_visible = false;

  [[nodiscard]] rund::kernel::NodeKind kind() const noexcept {
    return operation.kind();
  }
};

static_assert(sizeof(Operation) <= 512u,
              "active operation exceeded its inline footprint budget");
static_assert(sizeof(GraphCompileNode) <= 3072u,
              "graph compile node exceeded its footprint budget");

struct GraphCompileState {
  std::vector<std::vector<rund::kernel::GraphBufferRef>> buffer_storage{};
  std::vector<rund::kernel::GraphNode> node_storage{};
  std::map<std::uint64_t, std::uint64_t> logical_ids{};
  std::set<std::uint64_t> explicit_logical_ids{};
  std::map<std::uint64_t, std::uint64_t> logical_representatives{};
  std::uint64_t next_generated_logical_id = 1u;
  std::vector<rund::kernel::BufferRole> graph_roles{};
  std::vector<rund::AccelBufferDesc> graph_shapes{};
  std::vector<rund::GraphBufferVisibility> graph_visibilities{};
  std::vector<std::uint64_t> graph_alias_representatives{};
  std::vector<SourceStep> graph_binding_sources{};
  std::vector<std::uint64_t> graph_reset_bindings{};
  std::vector<GraphCompileNode> compile_nodes{};
  std::vector<rund::kernel::FusionNodePolicy> fusion_nodes{};
  std::vector<std::uint8_t> required_barriers{};
};

[[nodiscard]] bool
GraphShapeCanBeWalked(const rund::AccelGraph &graph) noexcept;

[[nodiscard]] bool
NodeBuffersCanBeWalked(const rund::AccelGraphNode &node) noexcept;

void ReserveGraphCompileState(GraphCompileState &state,
                              std::uint64_t node_count);

[[nodiscard]] bool ReserveExplicitGraphLogicalIds(const rund::AccelGraph &graph,
                                                  GraphCompileState &state);

[[nodiscard]] const char *
AppendGraphCompileNode(const ContextAdmission &admission,
                       const rund::AccelGraph &graph, std::uint64_t node_index,
                       GraphCompileState &state);

[[nodiscard]] rund::kernel::Graph KernelGraphFor(
    const GraphCompileState &state, rund::kernel::ComputeScalar scalar,
    rund::kernel::ComputeDomain domain,
    rund::kernel::ComputeFixedFormat fixed_format, const std::uint64_t *outputs,
    std::uint64_t output_count) noexcept;

} // namespace rund::node::accel::detail
