#include "../resource.hpp"

#include "../../../solve/shape.hpp"

#include <utility>

namespace rund::node::accel::detail {

rund::AccelCheck PrepareMetalSolve(const rund::AccelDevice &pick,
                                   const rund::kernel::SolveDesc &desc,
                                   const rund::kernel::SolvePlan &plan,
                                   const SolveBinds &bindings,
                                   std::shared_ptr<void> &out) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  out.reset();
  auto *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr || !SolveShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_solve_invalid"};
  }
  if (plan.element_bytes != 4u && plan.element_bytes != 8u) {
    return RejectElement(*adapter, "compute_solve_element_unsupported");
  }
  auto state = std::make_shared<MetalNumericPrepared>();
  state->adapter = adapter;
  const bool aux = plan.input == rund::kernel::SolveInput::Factor &&
                   plan.factor == rund::kernel::FactorOp::LU;
  MetalResidentReq requests[] = {
      {bindings.primary, bindings.primary_handle, &state->buffers[0]},
      {aux ? bindings.aux : bindings.status,
       aux ? bindings.aux_handle : bindings.status_handle, &state->buffers[1]},
      {bindings.rhs, bindings.rhs_handle, &state->buffers[2]},
      {bindings.output, bindings.output_handle, &state->buffers[3]},
      {bindings.status, bindings.status_handle, &state->buffers[4]},
  };
  rund::AccelCheck check = PreparedBuffers(pick, *state, requests);
  if (!check.ok) {
    return check;
  }
  check = PreparedPipeline(*state,
                           plan.element_bytes == 8u ? "rund_numeric_solve_i64"
                                                    : "rund_numeric_solve_i32",
                           FixedPolicy(plan.fixed_format));
  if (!check.ok) {
    return check;
  }
  state->params = NumericParams{static_cast<rund::kernel::u64>(plan.factor),
                                static_cast<rund::kernel::u64>(plan.layout),
                                plan.rows,
                                0u,
                                0u,
                                plan.batch_count,
                                plan.rhs_cols,
                                0u,
                                0u,
                                static_cast<rund::kernel::u32>(plan.input),
                                static_cast<rund::kernel::u32>(plan.pivot)};
  state->status_index = 4u;
  state->status_count = plan.status_count;
  state->status_primitive = rund::compute::detail::Primitive::Solve;
  state->semantic_status = true;
  return PublishPrepared(std::move(state), plan.batch_count, kMetalAlgebraLanes,
                         plan.pass_count, out);
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
