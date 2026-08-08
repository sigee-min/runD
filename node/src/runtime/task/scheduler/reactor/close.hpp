#pragma once

#include <rund/task/handle.hpp>

#include <cstdint>

namespace rund::node {

class Scheduler;

[[nodiscard]] ReasonCode ReactorCloseInvalidateFd(Scheduler &scheduler,
                                                  int fd) noexcept;

} // namespace rund::node
