#include "local.hpp"

#include "src/compute/backend.hpp"
#include "src/compute/pipeline/local.hpp"
#include "src/compute/pipeline/state.hpp"

#include <accel/check.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string_view>

namespace rund_node_test_pipeline {
namespace {

using rund::compute::Backend;
using rund::compute::Location;
using rund::compute::Reason;
using rund::compute::Status;
using rund::compute::detail::DeviceOps;
using rund::compute::detail::DeviceState;
using rund::compute::detail::PipelineState;
using rund::node::accel::detail::PreparedKernelPipeline;

struct SeedProbe final {
  const PreparedKernelPipeline *primary{};
  const PreparedKernelPipeline *alternate{};
  std::array<const PreparedKernelPipeline *, 2u> pipelines{};
  std::array<std::uint32_t, 2u> seeds{};
  std::size_t call_count{};
  std::size_t fail_call{std::numeric_limits<std::size_t>::max()};
  const char *failure_reason{"pipeline_generation_test_failure"};

  void reset(
      const std::size_t failed_call = std::numeric_limits<std::size_t>::max(),
      const char *const reason = "pipeline_generation_test_failure") noexcept {
    pipelines = {};
    seeds = {};
    call_count = 0u;
    fail_call = failed_call;
    failure_reason = reason;
  }
};

thread_local SeedProbe *active_probe = nullptr;

[[nodiscard]] rund::AccelCheck
RecordSeed(const PreparedKernelPipeline &pipeline,
           const std::uint32_t seed) noexcept {
  if (active_probe == nullptr || active_probe->call_count >= 2u) {
    return {false, "pipeline_generation_test_invalid"};
  }
  const std::size_t call = active_probe->call_count++;
  active_probe->pipelines[call] = &pipeline;
  active_probe->seeds[call] = seed;
  return call == active_probe->fail_call
             ? rund::AccelCheck{false, active_probe->failure_reason}
             : rund::AccelCheck{true, "ok"};
}

struct Fixture final {
  std::shared_ptr<DeviceState> device{std::make_shared<DeviceState>()};
  DeviceOps operations{};
  PipelineState state{};
  SeedProbe probe{};

  Fixture() noexcept {
    device->backend = Backend::Vulkan;
    device->ops = &operations;
    operations.seed_pipeline_generation = &RecordSeed;
    state.device = device;
    state.active_step_count = 1u;
    state.transactional = true;
    state.native_generation = 41u;
    state.native_parity = 1u;
    probe.primary = &state.prepared;
    probe.alternate = &state.alternate_prepared;
  }

  [[nodiscard]] Status seed(const std::uint64_t generation,
                            const std::uint8_t parity,
                            Location *const location = nullptr) noexcept {
    active_probe = &probe;
    const Status result = rund::compute::detail::seed_pipeline_generations(
        state, generation, parity, location);
    active_probe = nullptr;
    return result;
  }
};

[[nodiscard]] bool same_reason(const Location &location,
                               const std::string_view expected) noexcept {
  return location.native_reason_key != nullptr &&
         std::string_view{location.native_reason_key} == expected;
}

} // namespace

int CheckPipelineGenerationSeeding() {
  Fixture fixture;

  // A transactional generation-zero preparation selects the preceding bank
  // with deliberate uint32 wrap and the alternate bank at zero.
  fixture.probe.reset();
  Location location{};
  const Status initial = fixture.seed(0u, 0u, &location);
  if (!initial || fixture.probe.call_count != 2u ||
      fixture.probe.pipelines[0] != fixture.probe.primary ||
      fixture.probe.pipelines[1] != fixture.probe.alternate ||
      fixture.probe.seeds[0] != std::numeric_limits<std::uint32_t>::max() ||
      fixture.probe.seeds[1] != 0u || fixture.state.native_generation != 0u ||
      fixture.state.native_parity != 0u ||
      location.native_reason_key != nullptr) {
    return 1;
  }

  // The selected physical bank follows parity: parity zero uses the
  // preceding seed first, while parity one uses the current seed first.
  fixture.probe.reset();
  const Status even = fixture.seed(9u, 0u);
  if (!even || fixture.probe.call_count != 2u ||
      fixture.probe.seeds != std::array<std::uint32_t, 2u>{8u, 9u} ||
      fixture.state.native_generation != 9u ||
      fixture.state.native_parity != 0u) {
    return 2;
  }
  fixture.probe.reset();
  const Status odd = fixture.seed(9u, 1u);
  if (!odd || fixture.probe.call_count != 2u ||
      fixture.probe.seeds != std::array<std::uint32_t, 2u>{9u, 8u} ||
      fixture.state.native_generation != 9u ||
      fixture.state.native_parity != 1u) {
    return 3;
  }

  // A primary failure must not touch the alternate stream or commit state;
  // its opaque native key is projected only at the public boundary.
  fixture.state.native_generation = 17u;
  fixture.state.native_parity = 1u;
  fixture.probe.reset(0u, "pipeline_generation_primary_failure");
  location = {};
  const Status primary_failure = fixture.seed(3u, 0u, &location);
  if (primary_failure || primary_failure.reason() != Reason::PipelineInvalid ||
      fixture.probe.call_count != 1u ||
      fixture.probe.pipelines[0] != fixture.probe.primary ||
      fixture.probe.seeds[0] != 2u ||
      !same_reason(location, "pipeline_generation_primary_failure") ||
      fixture.state.native_generation != 17u ||
      fixture.state.native_parity != 1u) {
    return 4;
  }

  // An alternate failure is reached only after the primary succeeds and has
  // the same no-commit and native-key-preservation guarantees.
  fixture.probe.reset(1u, "pipeline_generation_alternate_failure");
  location = {};
  const Status alternate_failure = fixture.seed(3u, 0u, &location);
  if (alternate_failure ||
      alternate_failure.reason() != Reason::PipelineInvalid ||
      fixture.probe.call_count != 2u ||
      fixture.probe.pipelines[0] != fixture.probe.primary ||
      fixture.probe.pipelines[1] != fixture.probe.alternate ||
      fixture.probe.seeds != std::array<std::uint32_t, 2u>{2u, 3u} ||
      !same_reason(location, "pipeline_generation_alternate_failure") ||
      fixture.state.native_generation != 17u ||
      fixture.state.native_parity != 1u) {
    return 5;
  }

  return 0;
}

} // namespace rund_node_test_pipeline
