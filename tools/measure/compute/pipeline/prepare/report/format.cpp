#include "format.hpp"

#include "../../../suite/core.hpp"

#include <cstdio>

namespace rund::measure::compute::preparation_memory::report {

const char *
MemoryCategoryName(const ::rund::compute::MemoryCategory category) noexcept {
  using ::rund::compute::MemoryCategory;
  switch (category) {
  case MemoryCategory::Host:
    return "host";
  case MemoryCategory::Frame:
    return "frame";
  case MemoryCategory::Tile:
    return "tile";
  case MemoryCategory::Resident:
    return "resident";
  case MemoryCategory::Staging:
    return "staging";
  case MemoryCategory::Device:
    return "device";
  case MemoryCategory::Transfer:
    return "transfer";
  }
  return "unknown";
}

const char *MemoryUseName(const ::rund::compute::MemoryUse use) noexcept {
  using ::rund::compute::MemoryUse;
  switch (use) {
  case MemoryUse::Metadata:
    return "metadata";
  case MemoryUse::Input:
    return "input";
  case MemoryUse::PendingInput:
    return "pending_input";
  case MemoryUse::Output:
    return "output";
  case MemoryUse::Internal:
    return "internal";
  case MemoryUse::Scratch:
    return "scratch";
  case MemoryUse::Coordinator:
    return "coordinator";
  case MemoryUse::Traffic:
    return "traffic";
  }
  return "unknown";
}

void PrintUnsigned(const std::uint64_t value) {
  std::printf(",%llu", static_cast<unsigned long long>(value));
}

void PrintSize(const std::size_t value) {
  PrintUnsigned(static_cast<std::uint64_t>(value));
}

void PrintDecimal(const double value) { std::printf(",%.3f", value); }

void PrintCsvField(const std::string_view value) {
  std::putchar(',');
  ::rund::measure::compute::PrintCsv(value);
}

} // namespace rund::measure::compute::preparation_memory::report
