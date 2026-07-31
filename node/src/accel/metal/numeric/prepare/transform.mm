#include "../resource.hpp"

#include "../../../transform/shape.hpp"
#include "../../adapter.hpp"

#include <kernel/program/compute/transform/stage.hpp>

#include <utility>

namespace rund::node::accel::detail {

rund::AccelCheck PrepareMetalTransform(const rund::AccelDevice &pick,
                                       const rund::kernel::TransformDesc &desc,
                                       const rund::kernel::TransformPlan &plan,
                                       const TransformBinds &bindings,
                                       std::shared_ptr<void> &out) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  out.reset();
  auto *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr || !TransformShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_transform_invalid"};
  }
  if (bindings.input_real->element_bytes != 4u &&
      bindings.input_real->element_bytes != 8u) {
    return RejectElement(*adapter, "compute_transform_element_unsupported");
  }
  auto state = std::make_shared<MetalNumericPrepared>();
  state->adapter = adapter;
  MetalResidentReq requests[] = {
      {bindings.input_real, bindings.input_real_handle, &state->buffers[0]},
      {bindings.input_imag, bindings.input_imag_handle, &state->buffers[1]},
      {bindings.output_real, bindings.output_real_handle, &state->buffers[2]},
      {bindings.output_imag, bindings.output_imag_handle, &state->buffers[3]},
  };
  rund::AccelCheck check = PreparedBuffers(pick, *state, requests);
  if (!check.ok) {
    return check;
  }
  if (!PrepareTwiddle(*state, plan)) {
    SetMetalLastError(*adapter, "accel_metal_buffer_unavailable");
    return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
  }
  check = PreparedPipeline(*state,
                           bindings.input_real->element_bytes == 8u
                               ? "rund_numeric_transform_i64"
                               : "rund_numeric_transform_i32",
                           FixedPolicy(plan.fixed_format));
  if (!check.ok) {
    return check;
  }
  state->params =
      NumericParams{1u,
                    1u,
                    plan.element_count,
                    0u,
                    0u,
                    plan.normalization_divisor,
                    0u,
                    0u,
                    0u,
                    static_cast<rund::kernel::u32>(plan.direction),
                    static_cast<rund::kernel::u32>(plan.normalization)};
  const auto local =
      rund::kernel::transform_stage::Describe(plan.element_count, 1u);
  const rund::kernel::u64 groups = rund::kernel::transform_stage::Groups(
      rund::kernel::transform_stage::Threads(plan.element_count, local));
  state->transform_count = plan.element_count;
  return PublishPrepared(std::move(state), groups,
                         rund::kernel::transform_stage::Lanes, plan.pass_count,
                         out);
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)bindings;
  (void)out;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
