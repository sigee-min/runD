#include "backend/local.hpp"

#include "../../backend.hpp"
#include "../../exception.hpp"
#include "../../stats.hpp"
#include "../local.hpp"

#include <utility>

namespace rund::compute::detail {

Reason
project_pipeline_preparation_reason(const std::string_view reason) noexcept {
  const Reason boundary =
      reason ==
              node::accel::detail::PreparedPipelineTemplateStepCapacityReasonKey
          ? Reason::PipelineCapacity
          : Reason::LoweringInvalid;
  return project_reason(reason, boundary);
}

Status prepare_backend(PipelineState &value, Location &location) noexcept {
  location = {};
  if (value.device == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  if (value.device->backend == Backend::Cpu) {
    return Status::success();
  }
  try {
    const DeviceOps *const ops = value.device->ops;
    if (ops == nullptr || ops->prepare_pipeline == nullptr) {
      return Status::fail(Reason::AccelProgramInvalid);
    }
    auto primary = prepare_backend_stream(value, false, location);
    if (!primary) {
      return Status::fail(primary.reason());
    }
    value.prepared = std::move(primary).value();
    accumulate_run_facts(value.stats, value.prepared.preparation);
    if (value.transactional) {
      auto alternate = prepare_backend_stream(value, true, location);
      if (!alternate) {
        return Status::fail(alternate.reason());
      }
      value.alternate_prepared = std::move(alternate).value();
      accumulate_run_facts(value.stats, value.alternate_prepared.preparation);
    }
    return seed_pipeline_generations(value, 0u, 0u, &location);
  } catch (...) {
    compute_exception::rethrow_unless_capacity_exception();
    return Status::fail(Reason::PipelineCapacity);
  }
}

} // namespace rund::compute::detail
