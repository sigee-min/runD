#pragma once

#include <accel/check.hpp>

#include "../../kernel/callback.hpp"
#include "../state.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelCheck WaitCommand(MetalAdapter &adapter,
                                           void *command_buffer,
                                           rund::RuntimeStats *stats = nullptr);
[[nodiscard]] rund::AccelCheck QueueCommand(MetalAdapter &adapter,
                                            void *command_buffer,
                                            KernelCompletion completion,
                                            void *user) noexcept;

} // namespace rund::node::accel::detail
