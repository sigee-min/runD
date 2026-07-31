#include "../resource.hpp"

#include "../../../matrix/shape.hpp"

#include <kernel/program/compute/matrix/tile.hpp>

#include <utility>

namespace rund::node::accel::detail {

rund::AccelCheck PrepareMetalMatrix(const rund::AccelDevice &pick,
                                    const rund::kernel::MatrixDesc &desc,
                                    const rund::kernel::MatrixPlan &plan,
                                    const MatrixBinds &bindings,
                                    std::shared_ptr<void> &out) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  out.reset();
  auto *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr || !MatrixShapeOk(desc, plan, bindings)) {
    return rund::AccelCheck{false, "compute_matrix_invalid"};
  }
  if (plan.element_bytes != 4u && plan.element_bytes != 8u) {
    return RejectElement(*adapter, "compute_matrix_element_unsupported");
  }
  const rund::kernel::matrix_tile::Tiles tiles =
      rund::kernel::matrix_tile::Count(plan.rows, plan.cols, plan.batch_count);
  if (!tiles.ok) {
    return rund::AccelCheck{false, "compute_matrix_shape_overflow"};
  }
  auto state = std::make_shared<MetalNumericPrepared>();
  state->adapter = adapter;
  const bool transpose = plan.op == rund::kernel::MatrixOp::Transpose;
  MetalResidentReq requests[] = {
      {bindings.left, bindings.left_handle, &state->buffers[0]},
      {transpose ? bindings.output : bindings.right,
       transpose ? bindings.output_handle : bindings.right_handle,
       &state->buffers[1]},
      {bindings.output, bindings.output_handle, &state->buffers[2]},
  };
  rund::AccelCheck check = PreparedBuffers(pick, *state, requests);
  if (!check.ok) {
    return check;
  }
  check = PreparedPipeline(*state,
                           plan.element_bytes == 8u ? "rund_numeric_matrix_i64"
                                                    : "rund_numeric_matrix_i32",
                           MatrixPolicy(plan.arithmetic, plan.fixed_format));
  if (!check.ok) {
    return check;
  }
  state->params = NumericParams{static_cast<rund::kernel::u64>(plan.op),
                                static_cast<rund::kernel::u64>(plan.layout),
                                plan.rows,
                                plan.cols,
                                plan.inner,
                                plan.batch_count};
  state->params.mode = static_cast<rund::kernel::u32>(plan.arithmetic);
  return PublishPrepared(std::move(state), tiles.count,
                         rund::kernel::matrix_tile::Lanes, plan.pass_count,
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
