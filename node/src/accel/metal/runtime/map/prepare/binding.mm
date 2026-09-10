#include "local.hpp"

#include "../../../../kernel/backend/template/identity.hpp"
#include "../../../../kernel/step/map/stride.hpp"

namespace rund::node::accel::detail::metal_map_prepare {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
bool same_bindings(const MetalMapTemplateResources &prepared,
                   const rund::kernel::BindingSet &bindings) noexcept {
  if (prepared.input_strides.size() != bindings.resident_inputs.count ||
      prepared.output_strides.size() != bindings.resident_outputs.count ||
      prepared.input_strides.size() > 64u ||
      prepared.output_strides.size() > 64u) {
    return false;
  }
  const auto mask_fits = [](const std::uint64_t mask,
                            const std::size_t count) noexcept {
    return count == 64u || (mask >> count) == 0u;
  };
  if (!mask_fits(prepared.input_word_mask, prepared.input_strides.size()) ||
      !mask_fits(prepared.output_word_mask, prepared.output_strides.size())) {
    return false;
  }
  for (std::size_t index = 0u; index < prepared.input_strides.size(); ++index) {
    const auto *const ref = bindings.resident_inputs.ref(index);
    const bool word =
        (prepared.input_word_mask & (std::uint64_t{1u} << index)) != 0u;
    if (ref == nullptr || ref->stride_bytes != prepared.input_strides[index] ||
        MetalMapBindingWordAligned(*ref) != word) {
      return false;
    }
  }
  for (std::size_t index = 0u; index < prepared.output_strides.size();
       ++index) {
    const auto *const ref = bindings.resident_outputs.ref(index);
    const bool word =
        (prepared.output_word_mask & (std::uint64_t{1u} << index)) != 0u;
    if (ref == nullptr || ref->stride_bytes != prepared.output_strides[index] ||
        MetalMapBindingWordAligned(*ref) != word) {
      return false;
    }
  }
  return true;
}
#endif

} // namespace rund::node::accel::detail::metal_map_prepare

namespace rund::node::accel::detail {

bool MetalMapTemplateMatches(
    const MetalMapTemplateResources &prepared, const MetalAdapter &adapter,
    const rund::kernel::ComputePlan &plan,
    const rund::kernel::BindingSet &bindings) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  return prepared.adapter == &adapter &&
         backend_template_plan::same_plan(prepared.plan, plan) &&
         metal_map_prepare::same_bindings(prepared, bindings);
#else
  (void)prepared;
  (void)adapter;
  (void)plan;
  (void)bindings;
  return false;
#endif
}

} // namespace rund::node::accel::detail
