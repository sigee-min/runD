#include <accel/graph/node.hpp>

#include "local.hpp"

#include <kernel/program/compute/graph/schema.hpp>

#include <string_view>

namespace rund::node::accel::detail {

bool SameBindingName(const char *const lhs, const std::string &rhs) noexcept {
  return lhs != nullptr && std::string_view{lhs} == std::string_view{rhs};
}

bool BindingOrderOk(const rund::AccelGraphNode &node,
                    const rund::kernel::ExecutionMetadata &metadata) noexcept {
  const std::size_t data_count = metadata.binding_accesses.size();
  if (metadata.binding_accesses.size() != metadata.binding_names.size() ||
      data_count > static_cast<std::size_t>(node.buffer_count)) {
    return false;
  }
  const bool require_names =
      metadata.read_count > 1u || metadata.write_count > 1u;
  for (std::size_t index = 0u; index < data_count; ++index) {
    const std::size_t local = static_cast<std::size_t>(index);
    if (rund::kernel::ComputeAccessFor(node.buffers[index].role) !=
        metadata.binding_accesses[local]) {
      return false;
    }
    if ((require_names || node.buffers[index].binding_name != nullptr) &&
        !SameBindingName(node.buffers[index].binding_name,
                         metadata.binding_names[local])) {
      return false;
    }
  }
  if (!node.control.valid(node.buffer_count)) {
    return false;
  }
  const auto control_binding_ok =
      [&](const std::uint32_t local,
          const rund::kernel::GraphControlSource source,
          const std::uint64_t offset) {
        if (local == rund::kernel::kNoGraphControlBinding) {
          return source == rund::kernel::GraphControlSource::Descriptor;
        }
        if (local < data_count || local >= node.buffer_count) {
          return false;
        }
        const rund::AccelGraphBufferRef &ref = node.buffers[local];
        const rund::AccelBufferDesc shape = AccelGraphBufferShape(ref);
        const std::uint64_t width =
            source == rund::kernel::GraphControlSource::U64 ? 8u : 4u;
        return ref.role == rund::kernel::BufferRole::Read &&
               shape.scalar_width_bytes == width && shape.count != 0u &&
               offset % width == 0u && offset / width < shape.count;
      };
  if (node.control.has_count() &&
      !control_binding_ok(node.control.count_binding, node.control.count_source,
                          node.control.count_byte_offset)) {
    return false;
  }
  if (node.control.has_predicate() &&
      !control_binding_ok(node.control.predicate_binding,
                          node.control.predicate_source,
                          node.control.predicate_byte_offset)) {
    return false;
  }
  return true;
}

bool PrimitivePayloadOnly(const rund::AccelGraphNode &node,
                          const rund::kernel::NodeKind active) noexcept {
  const bool resident_window =
      active == rund::kernel::NodeKind::Window &&
      node.window.count_source != rund::kernel::ComputeCountSource::Descriptor;
  const bool window_control =
      resident_window && node.control.has_count() &&
      !node.control.has_predicate() && node.control.count_binding == 1u &&
      node.control.capacity == node.window.input_count &&
      node.control.iteration == 0u &&
      ((node.window.count_source ==
            rund::kernel::ComputeCountSource::BufferU32 &&
        node.control.count_source == rund::kernel::GraphControlSource::U32) ||
       (node.window.count_source ==
            rund::kernel::ComputeCountSource::BufferU64 &&
        node.control.count_source == rund::kernel::GraphControlSource::U64));
  if (active != rund::kernel::NodeKind::Map &&
      ((!resident_window &&
        (node.control.has_count() || node.control.has_predicate() ||
         node.control.capacity != 0u || node.control.iteration != 0u)) ||
       (resident_window && !window_control))) {
    return false;
  }
  if (active == rund::kernel::NodeKind::Map) {
    if (node.ir == nullptr) {
      return false;
    }
  } else if (node.ir != nullptr) {
    return false;
  }
  if (active != rund::kernel::NodeKind::Sort && node.sort.key_bits != 0u) {
    return false;
  }
  return (active == rund::kernel::NodeKind::Scan ||
          DefaultScanDescriptor(node.scan)) &&
         (active == rund::kernel::NodeKind::SegmentedScan ||
          DefaultSegmentedScanDescriptor(node.segmented_scan)) &&
         (active == rund::kernel::NodeKind::SegmentedReduce ||
          DefaultSegmentedReduceDescriptor(node.segmented_reduce)) &&
         (active == rund::kernel::NodeKind::Gather ||
          DefaultGatherDescriptor(node.gather)) &&
         (active == rund::kernel::NodeKind::Histogram ||
          DefaultHistogramDescriptor(node.histogram)) &&
         (active == rund::kernel::NodeKind::Partition ||
          DefaultPartitionDescriptor(node.partition)) &&
         (active == rund::kernel::NodeKind::Reduce ||
          DefaultReduceDescriptor(node.reduce)) &&
         (active == rund::kernel::NodeKind::Scatter ||
          DefaultScatterDescriptor(node.scatter)) &&
         (active == rund::kernel::NodeKind::ScatterReduce ||
          DefaultScatterReduceDescriptor(node.scatter_reduce)) &&
         (active == rund::kernel::NodeKind::Stencil ||
          DefaultStencilDescriptor(node.stencil)) &&
         (active == rund::kernel::NodeKind::Window ||
          DefaultWindowDescriptor(node.window)) &&
         (active == rund::kernel::NodeKind::Transform ||
          DefaultTransformDescriptor(node.transform)) &&
         (active == rund::kernel::NodeKind::Matrix ||
          DefaultMatrixDescriptor(node.matrix)) &&
         (active == rund::kernel::NodeKind::Factor ||
          DefaultFactorDescriptor(node.factor)) &&
         (active == rund::kernel::NodeKind::Solve ||
          DefaultSolveDescriptor(node.solve)) &&
         (active == rund::kernel::NodeKind::Spectrum ||
          DefaultSpectrumDescriptor(node.spectrum));
}

} // namespace rund::node::accel::detail
