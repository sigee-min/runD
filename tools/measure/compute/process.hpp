#pragma once

#include <sys/resource.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <cstdio>
#include <unistd.h>
#endif

#include <cstdint>
#include <limits>

namespace rund::measure::compute {

[[nodiscard]] constexpr std::uint64_t
RusageMaximumResidentBytes(const std::uint64_t reported,
                           const std::uint64_t unit_bytes) noexcept {
  return unit_bytes != 0u &&
                 reported >
                     std::numeric_limits<std::uint64_t>::max() / unit_bytes
             ? std::numeric_limits<std::uint64_t>::max()
             : reported * unit_bytes;
}

static_assert(RusageMaximumResidentBytes(7u, 1u) == 7u);
static_assert(RusageMaximumResidentBytes(7u, 1024u) == 7168u);
static_assert(RusageMaximumResidentBytes(
                  std::numeric_limits<std::uint64_t>::max(), 1024u) ==
              std::numeric_limits<std::uint64_t>::max());

[[nodiscard]] inline std::uint64_t ProcessMaximumResidentBytes() noexcept {
  rusage usage{};
  if (::getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) {
    return 0u;
  }
#if defined(__APPLE__)
  constexpr std::uint64_t unit_bytes = 1u;
#else
  constexpr std::uint64_t unit_bytes = 1024u;
#endif
  return RusageMaximumResidentBytes(static_cast<std::uint64_t>(usage.ru_maxrss),
                                    unit_bytes);
}

// Current RSS is paired with ru_maxrss so a preparation peak cannot be hidden
// by an unrelated, earlier compiler or setup high-water. Both implementations
// return physical resident bytes and fail closed to zero when the platform
// observation is unavailable.
[[nodiscard]] inline std::uint64_t ProcessCurrentResidentBytes() noexcept {
#if defined(__APPLE__)
  mach_task_basic_info_data_t info{};
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (::task_info(::mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info),
                  &count) != KERN_SUCCESS) {
    return 0u;
  }
  return static_cast<std::uint64_t>(info.resident_size);
#elif defined(__linux__)
  std::FILE *const file = std::fopen("/proc/self/statm", "r");
  if (file == nullptr) {
    return 0u;
  }
  unsigned long long total_pages = 0u;
  unsigned long long resident_pages = 0u;
  const int fields =
      std::fscanf(file, "%llu %llu", &total_pages, &resident_pages);
  std::fclose(file);
  const long reported_page_bytes = ::sysconf(_SC_PAGESIZE);
  if (fields != 2 || reported_page_bytes <= 0) {
    return 0u;
  }
  return RusageMaximumResidentBytes(
      static_cast<std::uint64_t>(resident_pages),
      static_cast<std::uint64_t>(reported_page_bytes));
#else
  return 0u;
#endif
}

[[nodiscard]] constexpr bool WithinAdditionalResidentBytes(
    const std::uint64_t current_before, const std::uint64_t current_after,
    const std::uint64_t maximum_before, const std::uint64_t maximum_after,
    const std::uint64_t additional_limit) noexcept {
  if (current_before == 0u || current_after == 0u || maximum_before == 0u ||
      maximum_after == 0u || maximum_after < maximum_before ||
      current_before >
          std::numeric_limits<std::uint64_t>::max() - additional_limit) {
    return false;
  }
  const std::uint64_t ceiling = current_before + additional_limit;
  return current_after <= ceiling &&
         (maximum_after == maximum_before || maximum_after <= ceiling);
}

static_assert(WithinAdditionalResidentBytes(100u, 110u, 200u, 200u, 10u));
static_assert(WithinAdditionalResidentBytes(100u, 105u, 110u, 120u, 20u));
static_assert(!WithinAdditionalResidentBytes(100u, 121u, 110u, 120u, 20u));
static_assert(!WithinAdditionalResidentBytes(100u, 105u, 110u, 121u, 20u));

[[nodiscard]] constexpr std::uint64_t
NonnegativeDelta(const std::uint64_t after,
                 const std::uint64_t before) noexcept {
  return after >= before ? after - before : 0u;
}

} // namespace rund::measure::compute
