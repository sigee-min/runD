#include "../model.hpp"

#include "../../../type.hpp"
#include "../../local.hpp"
#include "../../scan.hpp"

#include <kernel/program/compute/scan/identity.hpp>

#include <cstdint>
#include <vector>

namespace rund::compute::detail::graph_detail::describe_detail {

Status build_scan_node(const GraphState &state, const ScanStep &scan,
                                Draft &draft, graph::Node &info,
                                std::vector<kernel::GraphBufferRef> &refs,
                                kernel::GraphNode &canonical,
                                resource_detail::MemoryNode &memory) {
  refs.push_back({scan.input, kernel::BufferRole::Read});
  append_access(draft.description.info, info, scan.input,
                resource::AccessMode::Read);
  if (scan.count != 0u) {
    refs.push_back({scan.count, kernel::BufferRole::Read});
    append_access(draft.description.info, info, scan.count,
                  resource::AccessMode::Read);
  }
  refs.push_back({scan.output, kernel::BufferRole::Write});
  append_access(draft.description.info, info, scan.output,
                resource::AccessMode::Write);

  const auto operation = scan_operation(scan.operation);
  if (!operation) {
    return Status::fail(Reason::ScanOpUnsupported);
  }
  const auto element = scan_element(state.values[scan.input - 1u].type);
  if (!element) {
    return Status::fail(Reason::TypeUnsupported);
  }
  const kernel::ScanDesc desc{
      .op = *operation,
      .element = *element,
      .element_count = state.values[scan.input - 1u].count,
      .block_size = collective_block(state.values[scan.input - 1u].count),
      .count_source =
          scan.count == 0u
              ? kernel::ComputeCountSource::Descriptor
              : (type_bytes(state.values[scan.count - 1u].type) == 8u
                     ? kernel::ComputeCountSource::BufferU64
                     : kernel::ComputeCountSource::BufferU32)};
  const kernel::ScanHash hash = kernel::HashScan(desc);
  kernel::GraphControl control{};
  if (!scan.control.empty() || scan.control.iteration != 0u) {
    if (scan.count == 0u || scan.control.count != scan.count ||
        scan.control.predicate != 0u ||
        scan.control.capacity != state.values[scan.input - 1u].count ||
        scan.control.iteration == 0u) {
      return Status::fail(Reason::BoundedCountInvalid);
    }
    control = kernel::GraphControl{
        .count_source = state.values[scan.count - 1u].type == Type::U64
                            ? kernel::GraphControlSource::U64
                            : kernel::GraphControlSource::U32,
        .count_binding = 1u,
        .capacity = scan.control.capacity,
        .iteration = scan.control.iteration,
    };
    if (!control.valid(refs.size())) {
      return Status::fail(Reason::BoundedCountInvalid);
    }
  }

  info.operation = graph::Operation::Scan;
  info.elements = desc.element_count;
  info.footprint = graph::Footprint{
      .pattern = graph::AccessPattern::Prefix,
      .input_elements = desc.element_count,
      .output_elements = desc.element_count,
      .tile_elements = desc.block_size,
      .operation = static_cast<std::uint32_t>(desc.op),
  };
  canonical = kernel::GraphNode{.buffers = refs.data(),
                                .buffer_count = refs.size(),
                                .kind = kernel::NodeKind::Scan,
                                .primitive_hash_hi = hash.hi,
                                .primitive_hash_lo = hash.lo,
                                .element_count = desc.element_count,
                                .control = control};
  memory.domain = resource_detail::Domain{.count = scan.count};
  memory.write = memory.domain.empty() ? resource_detail::Write::Full
                                       : resource_detail::Write::Domain;
  return Status::success();
}


} // namespace rund::compute::detail::graph_detail::describe_detail
