#pragma once

#include <rund/task/handle.hpp>

#include <cstdint>

namespace rund::node {

class Scheduler;

[[nodiscard]] bool ReactorCloseInvalidateFd(Scheduler &scheduler, int fd,
                                            ReasonCode *failure) noexcept;

} // namespace rund::node
