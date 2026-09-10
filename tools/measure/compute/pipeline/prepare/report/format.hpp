#pragma once

#include "../model.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace rund::measure::compute::preparation_memory::report {

void PrintUnsigned(std::uint64_t value);
void PrintSize(std::size_t value);
void PrintDecimal(double value);
void PrintCsvField(std::string_view value);

[[nodiscard]] const char *
MemoryCategoryName(::rund::compute::MemoryCategory category) noexcept;
[[nodiscard]] const char *
MemoryUseName(::rund::compute::MemoryUse use) noexcept;

} // namespace rund::measure::compute::preparation_memory::report
