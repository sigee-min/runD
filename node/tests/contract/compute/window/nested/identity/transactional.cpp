#include "../../../pipeline/local.hpp"
#include "../../local.hpp"
#include "../identity.hpp"
#include "../local.hpp"

#include <node/runtime/compute/access.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <utility>

namespace rund::node::test_contract::window {

[[nodiscard]] int
CheckTransactionalBindingIdentity(rund::compute::Device &device,
                                  const rund::compute::Backend backend) {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 1u> initial{kOuterSeed};
  constexpr std::array<std::uint32_t, 1u> pending{kSentinel};
  auto increment =
      on(device)
          .map<std::uint32_t>("nested-binding-identity-increment", 1u,
                              [](auto value) { return value + 1u; })
          .compile();
  auto first_bank = device.upload<std::uint32_t>(initial);
  auto second_bank = device.upload<std::uint32_t>(pending);
  if (!increment || !first_bank || !second_bank) {
    return 1;
  }

  auto builder = pipeline(device);
  builder.state(*first_bank, *second_bank)
      .repeat<kInner>(*increment, read(*first_bank), write_final(*second_bank))
      .commit();
  const auto plan = builder.plan();
  if (!plan) {
    return 2;
  }
  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = plan->peak_bytes})
                      .prepare();
  if (!prepared) {
    std::fprintf(stderr, "nested transactional prepare backend=%u reason=%u\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned>(prepared.reason()));
    return 3;
  }
  const PipelineBindingIdentity frozen =
      CaptureBindingIdentity(*prepared, backend);
  if (!frozen.valid || !frozen.transactional || frozen.normal_jobs.empty() ||
      frozen.normal_jobs.size() != frozen.alternate_jobs.size()) {
    std::fprintf(stderr,
                 "nested transactional identity backend=%u valid=%u tx=%u "
                 "jobs=%llu/%llu\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned>(frozen.valid),
                 static_cast<unsigned>(frozen.transactional),
                 static_cast<unsigned long long>(frozen.normal_jobs.size()),
                 static_cast<unsigned long long>(frozen.alternate_jobs.size()));
    return 4;
  }

  std::array<std::uint32_t, 1u> first_value{};
  const Status first_run = prepared->run();
  const Status first_read = prepared->read(*second_bank, first_value);
  const bool first_identity =
      CaptureBindingIdentity(*prepared, backend) == frozen;
  constexpr std::uint32_t first_expected = kOuterSeed + kInner;
  if (!first_run || !first_read || !first_identity ||
      first_value[0u] != first_expected) {
    std::fprintf(stderr,
                 "nested transactional first backend=%u status=%u/%u "
                 "value=%u/%u identity=%u\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned>(first_run.ok()),
                 static_cast<unsigned>(first_run.reason()), first_value[0u],
                 first_expected, static_cast<unsigned>(first_identity));
    return 5;
  }

  std::array<std::uint32_t, 1u> second_value{};
  const Status second_run = prepared->run();
  const Stats stats = prepared->stats();
  const Status second_read = prepared->read(*second_bank, second_value);
  const bool second_identity =
      CaptureBindingIdentity(*prepared, backend) == frozen;
  constexpr std::uint32_t second_expected = kOuterSeed + 2u * kInner;
  if (!second_run || !second_read || !second_identity ||
      prepared->generation() != 2u || second_value[0u] != second_expected ||
      !WarmSetupClean(stats)) {
    std::fprintf(
        stderr,
        "nested transactional warm backend=%u status=%u/%u generation=%llu "
        "value=%u/%u identity=%u\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(second_run.ok()),
        static_cast<unsigned>(second_run.reason()),
        static_cast<unsigned long long>(prepared->generation()),
        second_value[0u], second_expected,
        static_cast<unsigned>(second_identity));
    return 6;
  }
  return 0;
}
} // namespace rund::node::test_contract::window
