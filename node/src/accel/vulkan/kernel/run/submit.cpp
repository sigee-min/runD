#include "../../../kernel/backend/execute.hpp"
#include "../../../kernel/backend/template_plan.hpp"
#include "../../../kernel/recurrence/plan.hpp"
#include "../../../kernel/status.hpp"
#include "../../../resident/window/admission/runtime/windows.hpp"

#include "../../collective/chunk.hpp"
#include "../../compact/local.hpp"
#include "../../descriptor.hpp"
#include "../../gather/local.hpp"
#include "../../histogram/local.hpp"
#include "../../kernel.hpp"
#include "../../map/api.hpp"
#include "../../map/local.hpp"
#include "../../map/source_upper.hpp"
#include "../../numeric/source.hpp"
#include "../../numeric/state.hpp"
#include "../../partition/local.hpp"
#include "../../range/local.hpp"
#include "../../reduce/local.hpp"
#include "../../scan/local.hpp"
#include "../../scan/source.hpp"
#include "../../scatter/local.hpp"
#include "../../scatter/reduce/model.hpp"
#include "../../segmented/local.hpp"
#include "../../segmented/reduce/model.hpp"
#include "../../sort/local/state.hpp"
#include "../manifest.hpp"
#include "../ops/prepare.hpp"
#include "../pipeline/capacity.hpp"
#include "../pipeline/evidence.hpp"
#include "../pipeline/recurrence.hpp"
#include "../pipeline/source.hpp"
#include "../pipeline/state.hpp"
#include "../reset_source.hpp"

#include "../../../primitive/block.hpp"
#include "../../../sort/block/vulkan.hpp"

#include <kernel/program/compute/scan/plan.hpp>

#include <limits>

namespace rund::node::accel::detail {

rund::AccelCheck SubmitPreparedVulkanKernel(
    const BackendRun &run, const std::shared_ptr<void> &prepared,
    const KernelCompletion completion, void *const user,
    PreparedMemoryMeter *const memory, const std::shared_ptr<void> &,
    const KernelTiming timing) noexcept {
  return run.pick != nullptr && prepared != nullptr && completion != nullptr
             ? SubmitVulkanResources(*run.pick, prepared, completion, user,
                                     memory, timing)
             : rund::AccelCheck{false, "accel_kernel_run_invalid"};
}

} // namespace rund::node::accel::detail
