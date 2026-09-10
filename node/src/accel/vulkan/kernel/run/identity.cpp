#include "../../../kernel/backend/execute.hpp"
#include "../../../kernel/backend/template/identity.hpp"
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
#include "../../map/source/upper.hpp"
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
#include "../reset/source.hpp"

#include "../../../primitive/block.hpp"
#include "../../../sort/block/vulkan.hpp"

#include <kernel/program/compute/scan/plan.hpp>

#include <limits>

namespace rund::node::accel::detail {

bool SameVulkanPipelineProgramTemplate(
    const KernelExecution &execution, const PreparedKernelProgramRoute &left,
    const PreparedKernelProgramRoute &right) noexcept {
  const std::uint64_t alignment =
      left.kernel == nullptr ? 0u : left.kernel->frozen_caps.storage_alignment;
  return backend_template_plan::same_program_template(execution, left, right,
                                                      alignment);
}

bool SameVulkanPipelineTemplate(const BackendRun &left,
                                const BackendRun &right) noexcept {
  const std::uint64_t alignment =
      left.pick == nullptr ? 0u : left.pick->caps.storage_alignment;
  const std::size_t left_reset_count =
      left.resets == nullptr ? 0u : left.resets->size();
  const std::size_t right_reset_count =
      right.resets == nullptr ? 0u : right.resets->size();
  return alignment != 0u && left_reset_count == right_reset_count &&
         backend_template_plan::same_template(left, right, alignment);
}

} // namespace rund::node::accel::detail
