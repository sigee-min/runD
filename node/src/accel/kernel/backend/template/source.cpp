#include "source.hpp"

#include "../../step/map/stride.hpp"

#include "../../../context/internal/execution.hpp"

namespace rund::node::accel::detail::backend_template_plan {

bool map_source_upper(const KernelExecutionStep &step,
                      const rund::kernel::ComputePlan &plan,
                      std::uint64_t &retained,
                      std::uint64_t &transient) noexcept {
  transient = 0u;
  return MapSpecializedSourceUpperBytes(step.artifact, plan, retained);
}

} // namespace rund::node::accel::detail::backend_template_plan
