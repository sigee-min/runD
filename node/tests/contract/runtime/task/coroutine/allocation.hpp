#pragma once

#include <cstdint>

namespace runtime_task_allocation {

void Start() noexcept;
void Stop() noexcept;
void FailNext() noexcept;
[[nodiscard]] std::uint64_t Count() noexcept;

} // namespace runtime_task_allocation
