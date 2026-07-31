#pragma once

#include "status.hpp"

#include <accel/runtime.hpp>

namespace rund::node::accel::detail {

inline void ProjectTelemetry(const PreparedPipelineControl &control,
                             rund::RuntimeStats &stats) noexcept {
  stats.generated_item_count = control.generated_item_count;
  stats.generated_capacity = control.generated_capacity;
  stats.indirect_dispatch_count = control.indirect_dispatch_count;
  stats.indirect_work_item_count = control.indirect_work_item_count;
  stats.iteration_count = control.iteration_count;
  stats.skipped_iteration_count = control.skipped_iteration_count;
  stats.conflict_count = control.conflict_count;
  stats.overflow_ordinal = control.overflow_ordinal;
}

} // namespace rund::node::accel::detail
