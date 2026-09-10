#include "local.hpp"

#include "../../../state/run.hpp"

#include <rund/counter.hpp>

namespace rund::compute::detail {

void record_cpu_algebra_status(CpuGraphRun &run, const Primitive primitive,
                               const std::uint64_t failed,
                               const std::uint32_t first) noexcept {
  if (failed == 0u || first == 0u) {
    return;
  }
  run.semantic_failure_count = ::rund::detail::counter::SaturatingAdd(
      run.semantic_failure_count, failed);
  if (run.semantic_status == 0u) {
    run.semantic_primitive = primitive;
    run.semantic_status = first;
  }
}

} // namespace rund::compute::detail
