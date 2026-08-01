#pragma once

#include <cstdint>

namespace node_compute_allocation {

void Start() noexcept;
void Stop() noexcept;
void FailNext() noexcept;
[[nodiscard]] std::uint64_t Count() noexcept;
[[nodiscard]] std::uint64_t Bytes() noexcept;

} // namespace node_compute_allocation
