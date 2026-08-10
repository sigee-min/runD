#pragma once

#include "state.hpp"

namespace rund::compute::detail {

[[nodiscard]] Status
validate_virtual_pipeline_program(const ProgramState &program,
                                  const VirtualBufferState &input,
                                  const VirtualBufferState &output) noexcept;

} // namespace rund::compute::detail
