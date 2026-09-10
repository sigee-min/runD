#include "../../../kernel/backend/execute.hpp"
#include "../../../kernel/backend/template/identity.hpp"
#include "../../kernel.hpp"

namespace rund::node::accel::detail {

bool SameMetalPipelineProgramTemplate(
    const KernelExecution &execution, const PreparedKernelProgramRoute &left,
    const PreparedKernelProgramRoute &right) noexcept {
  return backend_template_plan::same_program_template(execution, left, right,
                                                      1u);
}

bool SameMetalPipelineTemplate(const BackendRun &left,
                               const BackendRun &right) noexcept {
  return backend_template_plan::same_template(left, right, 1u);
}

} // namespace rund::node::accel::detail
