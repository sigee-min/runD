#include <accel/buffer.hpp>
#include <accel/check.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/context/value.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>
#include <accel/graph/visibility.hpp>

#include "local.hpp"

#include "../../context/admission.hpp"

#include <limits>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] std::uint64_t
LogicalIdFor(const std::uint64_t resident_id,
             std::map<std::uint64_t, std::uint64_t> &logical_ids,
             const std::set<std::uint64_t> &explicit_ids,
             std::uint64_t &next_generated_id) {
  if (resident_id == 0u) {
    return 0u;
  }
  const auto existing = logical_ids.find(resident_id);
  if (existing != logical_ids.end()) {
    return existing->second;
  }
  while (next_generated_id != 0u &&
         explicit_ids.find(next_generated_id) != explicit_ids.end()) {
    next_generated_id =
        next_generated_id == std::numeric_limits<std::uint64_t>::max()
            ? 0u
            : next_generated_id + 1u;
  }
  if (next_generated_id == 0u) {
    return 0u;
  }
  const std::uint64_t logical_id = next_generated_id;
  next_generated_id = logical_id == std::numeric_limits<std::uint64_t>::max()
                          ? 0u
                          : logical_id + 1u;
  logical_ids.emplace(resident_id, logical_id);
  return logical_id;
}

} // namespace

const char *
AppendGraphBufferRefs(const ContextAdmission &admission,
                      const rund::AccelGraphNode &node, const SourceStep source,
                      GraphCompileState &state, GraphCompileNode &compile_data,
                      std::vector<rund::kernel::GraphBufferRef> &buffers) {
  if (!source.valid()) {
    return "accel_kernel_graph_invalid";
  }
  compile_data.binding_indices.reserve(
      static_cast<std::size_t>(node.buffer_count));
  if (!compile_data.binding_indices.ok) {
    return "accel_kernel_graph_invalid";
  }
  for (std::uint64_t buffer_index = 0u; buffer_index < node.buffer_count;
       ++buffer_index) {
    const rund::AccelGraphBufferRef &ref = node.buffers[buffer_index];
    if (ref.visibility != rund::GraphBufferVisibility::External &&
        ref.visibility != rund::GraphBufferVisibility::Internal) {
      return "accel_kernel_graph_invalid";
    }
    if ((ref.init != rund::kernel::BufferInit::Preserve &&
         ref.init != rund::kernel::BufferInit::Zero) ||
        (ref.role == rund::kernel::BufferRole::Read &&
         ref.init != rund::kernel::BufferInit::Preserve)) {
      return "accel_kernel_graph_invalid";
    }
    const rund::AccelBufferDesc shape = AccelGraphBufferShape(ref);
    if (!AccelGraphBufferShapeValid(ref) || !CheckDesc(shape).ok) {
      return "accel_kernel_graph_invalid";
    }
    if (ref.buffer != nullptr) {
      const rund::AccelCheck buffer_check =
          ValidateAccelBufferForSupport(admission, *ref.buffer);
      if (!buffer_check.ok) {
        return "accel_kernel_buffer_owner_mismatch";
      }
    } else if (ref.logical_id == 0u) {
      return "accel_kernel_graph_invalid";
    }
    const std::uint64_t logical_id =
        ref.logical_id != 0u
            ? ref.logical_id
            : LogicalIdFor(ref.buffer->resident.id, state.logical_ids,
                           state.explicit_logical_ids,
                           state.next_generated_logical_id);
    if (logical_id == 0u) {
      return "accel_kernel_graph_invalid";
    }
    const std::uint64_t current_index =
        static_cast<std::uint64_t>(state.graph_roles.size());
    const auto [logical, new_logical] =
        state.logical_representatives.try_emplace(logical_id, current_index);
    const std::size_t representative =
        static_cast<std::size_t>(logical->second);
    if (!new_logical) {
      const rund::AccelBufferDesc &prior_shape =
          state.graph_shapes[representative];
      if (prior_shape.scalar_width_bytes != shape.scalar_width_bytes ||
          prior_shape.count != shape.count ||
          state.graph_visibilities[representative] != ref.visibility) {
        return "accel_kernel_graph_invalid";
      }
    }
    if (ref.role == rund::kernel::BufferRole::Write &&
        ref.visibility == rund::GraphBufferVisibility::External) {
      compile_data.fusion_write_visible = true;
    }
    buffers.push_back(rund::kernel::GraphBufferRef{
        .logical_id = logical_id,
        .role = ref.role,
        .init = ref.init,
    });
    state.graph_roles.push_back(ref.role);
    state.graph_shapes.push_back(rund::AccelBufferDesc{
        .scalar_width_bytes = shape.scalar_width_bytes,
        .count = shape.count,
        .usage = ref.role == rund::kernel::BufferRole::Read
                     ? rund::BufferUsage::ReadOnly
                     : rund::BufferUsage::WriteOnly,
    });
    state.graph_visibilities.push_back(ref.visibility);
    state.graph_alias_representatives.push_back(
        static_cast<std::uint64_t>(representative));
    state.graph_binding_sources.push_back(source);
    if (ref.init == rund::kernel::BufferInit::Zero) {
      state.graph_reset_bindings.push_back(current_index);
    }
    if (!compile_data.binding_indices.push_back(
            static_cast<std::uint64_t>(state.graph_roles.size() - 1u))) {
      return "accel_kernel_graph_invalid";
    }
  }
  return "ok";
}

} // namespace rund::node::accel::detail
