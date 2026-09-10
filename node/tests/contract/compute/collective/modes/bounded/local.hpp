#pragma once

#include "../local.hpp"

#include <array>
#include <cstdint>

namespace rund_node_collective_modes::bounded {

template <class T>
using ResidentOutput = rund::compute::Outputs<rund::compute::Bounded<T>,
                                              rund::compute::Bounded<T>,
                                              rund::compute::Bounded<T>,
                                              rund::compute::Bounded<T>,
                                              rund::compute::Bounded<T>,
                                              rund::compute::Bounded<T>>;

template <class T, class Count>
using ResidentProgram =
    rund::compute::Program<ResidentOutput<T>(T, Count)>;

using I32Program = ResidentProgram<std::int32_t, std::uint32_t>;
using U32Program = ResidentProgram<std::uint32_t, std::uint32_t>;
using I64Program = ResidentProgram<std::int64_t, std::uint64_t>;
using U64Program = ResidentProgram<std::uint64_t, std::uint64_t>;
using Fixed32Program =
    ResidentProgram<rund::compute::Fixed<16, 16>, std::uint32_t>;
using Fixed64Program =
    ResidentProgram<rund::compute::Fixed<20, 44>, std::uint64_t>;

struct ResidentRangeIdentity final {
  std::uint64_t hi{};
  std::uint64_t lo{};

  [[nodiscard]] friend constexpr bool
  operator==(const ResidentRangeIdentity &,
             const ResidentRangeIdentity &) = default;
};

struct ResidentRangeFreeze final {
  std::array<ResidentRangeIdentity, 6u> source{};
  std::array<ResidentRangeIdentity, 6u> execution{};

  [[nodiscard]] friend constexpr bool
  operator==(const ResidentRangeFreeze &, const ResidentRangeFreeze &) =
      default;
};

[[nodiscard]] bool CheckResidentPlan(const rund::compute::Backend backend,
                                     const Domain domain,
                                     const I32Program &program,
                                     ResidentRangeFreeze &freeze);
[[nodiscard]] bool CheckResidentPlan(const rund::compute::Backend backend,
                                     const Domain domain,
                                     const U32Program &program,
                                     ResidentRangeFreeze &freeze);
[[nodiscard]] bool CheckResidentPlan(const rund::compute::Backend backend,
                                     const Domain domain,
                                     const I64Program &program,
                                     ResidentRangeFreeze &freeze);
[[nodiscard]] bool CheckResidentPlan(const rund::compute::Backend backend,
                                     const Domain domain,
                                     const U64Program &program,
                                     ResidentRangeFreeze &freeze);
[[nodiscard]] bool CheckResidentPlan(const rund::compute::Backend backend,
                                     const Domain domain,
                                     const Fixed32Program &program,
                                     ResidentRangeFreeze &freeze);
[[nodiscard]] bool CheckResidentPlan(const rund::compute::Backend backend,
                                     const Domain domain,
                                     const Fixed64Program &program,
                                     ResidentRangeFreeze &freeze);

[[nodiscard]] bool CheckResident(const rund::compute::Backend backend,
                                 Domain domain);
[[nodiscard]] bool CheckAggregate(rund::compute::Backend backend,
                                  DomainEvidence &evidence, Domain domain);
[[nodiscard]] bool CheckWindow(rund::compute::Backend backend,
                               DomainEvidence &evidence, Domain domain);

} // namespace rund_node_collective_modes::bounded
