#pragma once

#include <cstdint>

namespace kernel_contract_test::memory_allocation {

void Reset() noexcept;
void Stop() noexcept;
void FailNext() noexcept;
std::uint64_t Count() noexcept;

} // namespace kernel_contract_test::memory_allocation
