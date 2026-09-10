#include "../../../accel/kernel/prepared/run.hpp"
#include "internal.hpp"
#include "result.hpp"

#include "../local.hpp"
#include "../state.hpp"
#include "../../backend.hpp"

namespace rund::compute::detail {

[[nodiscard]] PipelineOutcome
run_accel(PipelineState &state,
          const node::accel::detail::PipelineSubmitMode mode) {
  PipelineOutcome outcome{};
  const DeviceOps *const ops = state.device->ops;
  if (ops == nullptr || ops->run_pipeline == nullptr) {
    outcome.status = Status::fail(Reason::AccelProgramInvalid);
    return outcome;
  }
  const std::size_t active = state.active_step_count;
  if (active == 0u) {
    node::accel::detail::PreparedPipelineEvidence empty{};
    empty.check = {true, "ok"};
    empty.shared.identity.backend = state.device->backend == Backend::Metal
                                        ? rund::AccelApi::Metal
                                        : rund::AccelApi::Vulkan;
    empty.shared.outcome.ok = true;
    empty.shared.outcome.reason = "ok";
    return finish_accel_pipeline(state, empty);
  }
  const node::accel::detail::PreparedKernelPipeline &prepared =
      state.transactional && state.attempt.parity != 0u
          ? state.alternate_prepared
          : state.prepared;
  if (!prepared.ok) {
    outcome.status = Status::fail(Reason::PipelineInvalid);
    return outcome;
  }

  return finish_accel_pipeline(
      state, ops->run_pipeline(*state.device, prepared, mode));
}

} // namespace rund::compute::detail
