#include "local.hpp"

namespace rund::compute::detail {

VirtualRunDispatchResult
dispatch_result(const VirtualExecutionResult &execution) noexcept {
  return VirtualRunDispatchResult{
      .status = execution.status,
      .failed_page = execution.failed_page,
      .output_hash = execution.output_hash,
      .poison_pipeline = execution.poison_pipeline,
      .selected = true,
      .direct_terminal = true,
      .certainty = execution.certainty.value_or(
          execution.poison_pipeline ? VirtualRunWriteCertainty::UnknownMayWrite
                                    : VirtualRunWriteCertainty::KnownNoWrite),
  };
}

} // namespace rund::compute::detail

namespace rund::compute::detail::virtual_run_dispatch {

VirtualRunWriteCertainty epoch_certainty(const bool poison_pipeline) noexcept {
  return poison_pipeline ? VirtualRunWriteCertainty::UnknownMayWrite
                         : VirtualRunWriteCertainty::KnownNoWrite;
}

} // namespace rund::compute::detail::virtual_run_dispatch
