#include "internal.hpp"

#include <new>
#include <utility>
#include <vector>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck ProjectMetalView(const rund::AccelDevice &pick,
                                  const BoundStep &source,
                                  const KernelPreparationMode mode,
                                  const KernelViewLayout *const views,
                                  const RunBinds *const view_binds,
                                  std::shared_ptr<MetalViewLowering> &out) {
  out.reset();
  if (!MetalViewRequiresLowering(source)) {
    return rund::AccelCheck{true, "ok"};
  }
  if (!ValidateMetalViewSource(source)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  const RunBinds &original = *source.source_binds;
  std::shared_ptr<MetalViewLowering> view;
  try {
    view = std::make_shared<MetalViewLowering>();
    view->transfers.reserve(source.step->graph_binding_indices.size());
    view->transfer_by_binding.resize(original.size(), 0u);
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  std::vector<MetalViewReplacement> replacements;
  std::vector<std::uint32_t> replacement_by_binding;
  try {
    replacements.reserve(source.step->graph_binding_indices.size());
    replacement_by_binding.resize(original.size(), 0u);
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  for (std::size_t local = 0u;
       local < source.step->graph_binding_indices.size(); ++local) {
    const std::uint64_t index = source.step->graph_binding_indices[local];
    if (index >= original.size() ||
        replacement_by_binding[static_cast<std::size_t>(index)] != 0u) {
      continue;
    }
    const rund::kernel::ResidentBufferRef &ref = original.refs()[index];
    bool normalize_singleton = false;
    if (!MetalViewReferenceNeedsLowering(ref, normalize_singleton)) {
      continue;
    }
    MetalResidentBufferResult external =
        ResolveMetalViewExternal(pick, ref, original.handles()[index]);
    if (!external.check.ok) {
      return external.check;
    }
    if (normalize_singleton) {
      external.ref.stride_bytes = external.ref.element_bytes;
      try {
        replacements.push_back(
            MetalViewReplacement{.resident = std::move(external)});
        replacement_by_binding[static_cast<std::size_t>(index)] =
            static_cast<std::uint32_t>(replacements.size());
      } catch (const std::bad_alloc &) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      continue;
    }
    bool planned = false;
    MetalResidentBufferResult dense = ResolveMetalViewDense(
        pick, index, ref, mode, views, view_binds, planned);
    if (!dense.check.ok) {
      return dense.check;
    }
    try {
      view->transfers.push_back(MetalViewTransfer{
          .binding = index,
          .external = std::move(external),
          .dense = dense,
          .count = ref.count,
          .element_bytes = ref.element_bytes,
          .offset_bytes = ref.offset_bytes,
          .stride_bytes = ref.stride_bytes,
          .input = ref.usage == rund::kernel::kResidentUsageRead,
          .planned = planned,
      });
      view->has_input = view->has_input || view->transfers.back().input;
      replacements.push_back(
          MetalViewReplacement{.resident = std::move(dense)});
      view->transfer_by_binding[static_cast<std::size_t>(index)] =
          static_cast<std::uint32_t>(view->transfers.size());
      replacement_by_binding[static_cast<std::size_t>(index)] =
          static_cast<std::uint32_t>(replacements.size());
    } catch (const std::bad_alloc &) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  if (replacements.empty()) {
    return rund::AccelCheck{true, "ok"};
  }
  const rund::AccelCheck bound = BindMetalViewArguments(
      original, replacements, replacement_by_binding, *view);
  if (!bound.ok) {
    return bound;
  }
  if (!view->binds.valid() ||
      !RebindBoundStep(source, view->binds, view->step)) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  out = std::move(view);
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
