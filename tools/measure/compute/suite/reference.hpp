#pragma once

#include "core.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace rund::measure::compute {

struct HashEvidence final {
  std::uint64_t graph = 0u;
  std::uint64_t output = 0u;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return graph != 0u && output != 0u;
  }

  [[nodiscard]] constexpr bool
  operator==(const HashEvidence &) const noexcept = default;
};

struct ReferenceKey final {
  std::string_view workload{};
  std::string_view variant{};
  std::size_t count = 0u;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return !workload.empty() && !variant.empty();
  }

  [[nodiscard]] constexpr bool
  operator==(const ReferenceKey &) const noexcept = default;
};

enum class ReferenceStatus : std::uint8_t {
  Established,
  Matched,
  Missing,
  Mismatch,
  Full,
  Invalid,
};

template <std::size_t Capacity> struct ReferenceLedger final {
  struct Entry final {
    ReferenceKey key{};
    HashEvidence evidence{};
  };

  std::array<Entry, Capacity> entries{};
  std::size_t size = 0u;

  [[nodiscard]] constexpr ReferenceStatus
  check(const bool establish, const ReferenceKey key,
        const HashEvidence evidence) noexcept {
    if (!key.valid() || !evidence.valid()) {
      return ReferenceStatus::Invalid;
    }
    for (std::size_t index = 0u; index < size; ++index) {
      if (entries[index].key == key) {
        return entries[index].evidence == evidence ? ReferenceStatus::Matched
                                                   : ReferenceStatus::Mismatch;
      }
    }
    if (!establish) {
      return ReferenceStatus::Missing;
    }
    if (size == Capacity) {
      return ReferenceStatus::Full;
    }
    entries[size++] = Entry{.key = key, .evidence = evidence};
    return ReferenceStatus::Established;
  }
};

consteval bool ReferenceLedgerContract() {
  ReferenceLedger<2u> ledger{};
  constexpr ReferenceKey first{"family", "map", 4096u};
  constexpr ReferenceKey second{"collective", "scan", 262144u};
  constexpr ReferenceKey third{"fixed", "wide", 4096u};
  constexpr HashEvidence evidence{11u, 13u};
  return ledger.check(true, first, evidence) == ReferenceStatus::Established &&
         ledger.check(false, first, evidence) == ReferenceStatus::Matched &&
         ledger.check(false, first, HashEvidence{11u, 17u}) ==
             ReferenceStatus::Mismatch &&
         ledger.check(false, second, evidence) == ReferenceStatus::Missing &&
         ledger.check(true, second, evidence) == ReferenceStatus::Established &&
         ledger.check(true, third, evidence) == ReferenceStatus::Full &&
         ledger.size == 2u;
}

static_assert(ReferenceLedgerContract());

inline constexpr std::size_t ReferenceCapacity = 64u;
inline constexpr std::size_t RequiredReferenceEntries = 37u;
static_assert(ReferenceCapacity >= RequiredReferenceEntries);

// One process-wide ledger establishes CPU references and checks native parity.
// Its storage is defined by reference.cpp so each TU shares one authority.
extern ReferenceLedger<ReferenceCapacity> references;
extern HashEvidence batch_reference;

[[nodiscard]] bool CheckReference(Backend backend, ReferenceKey key,
                                  HashEvidence evidence);

} // namespace rund::measure::compute
