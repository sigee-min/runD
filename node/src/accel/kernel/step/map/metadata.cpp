#include "local.hpp"

namespace rund::node::accel::detail {

const rund::kernel::ExecutionMetadata &
MapMetadata(const KernelExecutionStep &step) noexcept {
  return step.artifact.metadata;
}

} // namespace rund::node::accel::detail
