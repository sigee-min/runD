#include "strict/local.hpp"

namespace rund::kernel::program_detail {

void AttachStrictFloatTelemetry(Telemetry& telemetry,
                                const FoldOperation operation,
                                const StrictFloatReductionPolicy policy,
                                const WorkerBackendCapabilities& capabilities) {
  telemetry.strict_fp_software_reference =
      FoldOperationAllowsFloatingPoint(operation) &&
      StrictFloatReductionValid(operation, policy);
  telemetry.strict_fp_backend_supported = capabilities.supports_strict_fp_fold;
}

} // namespace rund::kernel::program_detail
