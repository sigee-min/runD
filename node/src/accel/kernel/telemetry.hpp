#pragma once

#include "status.hpp"

#include <accel/runtime.hpp>

namespace rund::node::accel::detail {

inline void ProjectTelemetry(const PreparedPipelineControl &control,
                             rund::RuntimeStats &stats) noexcept {
  stats.run.work.generated_item_count = control.generated_item_count;
  stats.run.work.generated_capacity = control.generated_capacity;
  stats.run.work.indirect_dispatch_count = control.indirect_dispatch_count;
  stats.run.work.indirect_work_item_count = control.indirect_work_item_count;
  stats.run.work.iteration_count = control.iteration_count;
  stats.run.work.skipped_iteration_count = control.skipped_iteration_count;
  stats.run.work.conflict_count = control.conflict_count;
  stats.run.work.overflow_ordinal = control.overflow_ordinal;
}

} // namespace rund::node::accel::detail
