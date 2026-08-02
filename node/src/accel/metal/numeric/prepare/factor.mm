#include "../resource.hpp"

#include "../../../factor/shape.hpp"

#include <utility>

namespace rund::node::accel::detail {

rund::AccelCheck PrepareMetalFactor(const rund::AccelDevice &pick,
                                    const rund::kernel::FactorDesc &desc,
                                    const rund::kernel::FactorPlan &plan,
                                    const FactorBinds &bindings,
                                    std::shared_ptr<void> &out,
                                    const MetalKernelImmutablePipelines *const
                                        pipelines) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  out.reset();
  auto *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr || !FactorShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_factor_invalid"};
  }
  if (plan.element_bytes != 4u && plan.element_bytes != 8u) {
    return RejectElement(*adapter, "compute_factor_element_unsupported");
  }
  auto state = std::make_shared<MetalNumericPrepared>();
  state->adapter = adapter;
  const bool lu = plan.op == rund::kernel::FactorOp::LU;
  MetalResidentReq requests[] = {
      {bindings.input, bindings.input_handle, &state->buffers[0]},
      {bindings.factor, bindings.factor_handle, &state->buffers[1]},
      {lu ? bindings.aux : bindings.status,
       lu ? bindings.aux_handle : bindings.status_handle, &state->buffers[2]},
      {bindings.status, bindings.status_handle, &state->buffers[3]},
  };
  rund::AccelCheck check = PreparedBuffers(pick, *state, requests);
  if (!check.ok) {
    return check;
  }
  check = PreparedPipeline(*state,
                           plan.element_bytes == 8u ? "rund_numeric_factor_i64"
                                                    : "rund_numeric_factor_i32",
                           FixedPolicy(plan.fixed_format), pipelines);
  if (!check.ok) {
    return check;
  }
  state->params = NumericParams{static_cast<rund::kernel::u64>(plan.op),
                                static_cast<rund::kernel::u64>(plan.layout),
                                plan.rows,
                                plan.cols,
                                0u,
                                plan.batch_count,
                                0u,
                                plan.factor_count / plan.batch_count,
                                0u,
                                static_cast<rund::kernel::u32>(plan.output),
                                static_cast<rund::kernel::u32>(plan.pivot)};
  state->status_index = 3u;
  state->status_count = plan.status_count;
  state->status_primitive = rund::compute::detail::Primitive::Factor;
  state->semantic_status = true;
  return PublishPrepared(std::move(state), plan.batch_count, kMetalAlgebraLanes,
                         plan.pass_count, out);
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)bindings;
  (void)out;
  (void)pipelines;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
