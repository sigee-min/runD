#include "reuse/local.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <span>

int RunComputeReuseContract() {
  using namespace rund_node_test_compute_reuse;
  if (!CheckTelemetryMath()) {
    return 1;
  }
  auto device = rund::compute::open(rund::compute::Target::cpu());
  if (!device) {
    return 2;
  }
  auto program = rund::compute::on(device.value())
                     .map<std::int32_t>(
                         "reuse", 4, [](auto value) { return value * 2 + 5; })
                     .compile();
  if (!program) {
    return 2;
  }

  const std::array<std::int32_t, 4> input{1, 2, 3, 4};
  if (!CheckInitialHostReuse(program.value(),
                             std::span<const std::int32_t>{input})) {
    return 10;
  }
  if (!CheckProgramConcurrency(device.value())) {
    std::fprintf(stderr, "program convenience execution is not isolated\n");
    std::fprintf(stderr, "program convenience execution is not isolated\n");
    return 12;
  }
  if (!CheckProgramLifetime(device.value())) {
    std::fprintf(stderr, "program/run ownership contract failed\n");
    return 13;
  }

  using Q16_16 = rund::compute::Fixed<16u, 16u>;
  using Q20_44 = rund::compute::Fixed<20u, 44u>;
  static_assert(sizeof(Q16_16) == sizeof(std::uint32_t));
  static_assert(sizeof(Q20_44) == sizeof(std::uint64_t));
  if (!CheckHostIdentities(device.value())) {
    return 11;
  }
  if (const int one_shot = CheckReadOnlyOneShot(device.value());
      one_shot != 0) {
    return 20 + one_shot;
  }
  return CheckBufferReuseAndReceipt(device.value(), program.value(),
                                    std::span<const std::int32_t>{input});
}
