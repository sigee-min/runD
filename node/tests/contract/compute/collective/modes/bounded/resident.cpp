#include "support.hpp"

#include <array>
#include <cstdio>
#include <vector>

namespace rund_node_collective_modes::bounded {
namespace {

template <class T>
[[nodiscard]] bool CheckResidentBoundedWindow(
    const rund::compute::Backend backend, const Domain domain) {
  using namespace rund::compute;
  using Count = CountFor<T>;
  auto program = BuildResidentWindowProgram<T>(backend);
  if (!program) {
    std::fprintf(stderr,
                 "compute resident bounded window compile width=%zu fixed=%u "
                 "reason=%.*s\n",
                 sizeof(T), detail::FixedValue<T> ? 1u : 0u,
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return false;
  }

  ResidentRangeFreeze initial_freeze{};
  if (backend == Backend::Cpu &&
      !CheckResidentPlan(backend, domain, *program, initial_freeze)) {
    std::fprintf(stderr,
                 "compute resident bounded window frozen plan width=%zu "
                 "fixed=%u\n",
                 sizeof(T), detail::FixedValue<T> ? 1u : 0u);
    return false;
  }
  if (backend != Backend::Cpu &&
      !CheckResidentPlan(backend, domain, *program, initial_freeze)) {
    std::fprintf(stderr,
                 "compute resident bounded window accel plan width=%zu "
                 "fixed=%u backend=%u\n",
                 sizeof(T), detail::FixedValue<T> ? 1u : 0u,
                 static_cast<unsigned>(backend));
    return false;
  }

  std::vector<T> full(kResidentWindowCapacity);
  std::vector<T> small(kResidentWindowCapacity);
  std::vector<T> zero(kResidentWindowCapacity, Zero<T>());
  for (std::size_t index = 0u; index < kResidentWindowCapacity; ++index) {
    full[index] = ResidentPattern<T>(index);
    small[index] = index < kResidentWindowSmallCount
                       ? ResidentPattern<T>(index + 3u)
                       : ResidentPoison<T>(index);
  }
  const std::array<Count, 1u> full_count{
      static_cast<Count>(kResidentWindowCapacity)};
  const std::array<Count, 1u> small_count{
      static_cast<Count>(kResidentWindowSmallCount)};
  const std::array<Count, 1u> zero_count{Count{0}};
  const std::array<Count, 1u> overflow_count{
      static_cast<Count>(kResidentWindowCapacity + 1u)};

  auto job = program->resident(full, full_count);
  if (!job || !job->run() ||
      !ResidentOutputMatches(job->read_all(),
                             ResidentWindowOracle(full, full.size())) ||
      !job->write(small, small_count) || !job->run()) {
    std::fprintf(stderr,
                 "compute resident bounded window full/small execution "
                 "width=%zu fixed=%u\n",
                 sizeof(T), detail::FixedValue<T> ? 1u : 0u);
    return false;
  }
  const auto small_expected =
      ResidentWindowOracle(small, kResidentWindowSmallCount);
  auto small_output = job->read_all();
  if (!ResidentOutputMatches(small_output, small_expected)) {
    std::fprintf(stderr,
                 "compute resident bounded window poisoned-tail mismatch "
                 "width=%zu fixed=%u\n",
                 sizeof(T), detail::FixedValue<T> ? 1u : 0u);
    return false;
  }

  const Status overflow = job->write(full, overflow_count);
  auto retained = job->read_all();
  if (overflow || overflow.error() != "compute_workset_overflow" ||
      !ResidentOutputMatches(retained, small_expected) ||
      !job->write(zero, zero_count) || !job->run()) {
    std::fprintf(stderr,
                 "compute resident bounded window overflow/zero width=%zu "
                 "fixed=%u reason=%.*s\n",
                 sizeof(T), detail::FixedValue<T> ? 1u : 0u,
                 static_cast<int>(overflow.error().size()),
                 overflow.error().data());
    return false;
  }
  const auto empty_expected = ResidentWindowOracle(zero, 0u);
  if (!ResidentOutputMatches(job->read_all(), empty_expected) ||
      !job->write(full, full_count) || !job->run() ||
      !ResidentOutputMatches(job->read_all(),
                             ResidentWindowOracle(full, full.size()))) {
    std::fprintf(stderr,
                 "compute resident bounded window zero/retry width=%zu "
                 "fixed=%u\n",
                 sizeof(T), detail::FixedValue<T> ? 1u : 0u);
    return false;
  }

  ResidentRangeFreeze final_freeze{};
  return backend != Backend::Cpu ||
         (CheckResidentPlan(backend, domain, *program, final_freeze) &&
          initial_freeze == final_freeze);
}

} // namespace

bool CheckResident(const rund::compute::Backend backend, const Domain domain) {
  switch (domain) {
  case Domain::I32:
    return CheckResidentBoundedWindow<std::int32_t>(backend, domain);
  case Domain::U32:
    return CheckResidentBoundedWindow<std::uint32_t>(backend, domain);
  case Domain::I64:
    return CheckResidentBoundedWindow<std::int64_t>(backend, domain);
  case Domain::U64:
    return CheckResidentBoundedWindow<std::uint64_t>(backend, domain);
  case Domain::Fixed16x16:
    return CheckResidentBoundedWindow<rund::compute::Fixed<16, 16>>(backend,
                                                                     domain);
  case Domain::Fixed20x44:
    return CheckResidentBoundedWindow<rund::compute::Fixed<20, 44>>(backend,
                                                                     domain);
  case Domain::Lane32:
  case Domain::Lane64:
    return false;
  }
  return false;
}

} // namespace rund_node_collective_modes::bounded
