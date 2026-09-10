#include "local.hpp"

#include "backing.hpp"
#include "golden.hpp"
#include "model.hpp"

#include "../../../target/selection.hpp"

#include "src/compute/backend.hpp"
#include "src/compute/device/state.hpp"
#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/state.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)
#include "src/accel/kernel/fault.hpp"

#include <node/accel/buffer.hpp>
#endif

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstdio>
#include <memory>
#include <span>

namespace rund_node_test_virtual::product {

int CheckProductDeviceVsmUnknown(const rund::compute::Backend backend) {
#if defined(RUND_NODE_TEST_BACKEND_CPU)
  return backend == rund::compute::Backend::Cpu ? 0 : 1;
#else
  using namespace rund::compute;
  if (backend == Backend::Cpu) {
    return 0;
  }
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    return 1;
  }
  auto program =
      on(*opened)
          .map<std::int32_t>("virtual-product-device-vsm-unknown", PageElements,
                             [](auto value) { return (value + 5) * 3; })
          .compile();
  auto input_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  auto output_backing =
      std::make_shared<MemoryVirtualBacking>(LogicalBytes, ElementPageBytes);
  std::array<std::int32_t, LogicalElements> seeded{};
  SeedInput(seeded);
  auto input = virtual_buffer<std::int32_t>(LogicalElements, input_backing);
  auto output = virtual_buffer<std::int32_t>(LogicalElements, output_backing);
  auto prepared =
      program && input && output
          ? virtual_pipeline(*program, *input, *output, ResidencyConfig{})
          : Result<VirtualPipeline<std::int32_t(std::int32_t)>>::fail(
                Reason::PipelineInvalid);
  if (!program || !input_backing->seed(std::as_bytes(std::span{seeded})) ||
      !prepared) {
    return 2;
  }
  const auto &state = detail::VirtualPipelineAccess::state(*prepared);
  if (state == nullptr) {
    return 2;
  }
  // This is the explicit DeviceVsm unknown-device contract. The ordinary
  // callback-backed default now declines DeviceVsm before backend preparation;
  // retain the fault-injection proof by opting this fixture in explicitly.
  state->geometry.device_vsm_required = true;
  if (!prepared->run()) {
    return 2;
  }
  const Stats cold = prepared->stats();
  const BackingFacts published = output_backing->facts();
  const std::uint64_t version =
      detail::VirtualBackingAccess::version(*output_backing);
  if (cold.command_submits != 1u || cold.dispatches != 1u ||
      cold.final_dispatches != 1u ||
      cold.pipeline.residency.window_handoff_count != 1u ||
      cold.pipeline.residency.window_queue_call_count != 1u ||
      cold.pipeline.residency.page_out_count != PageCount ||
      published.write_count != PageCount ||
      detail::VirtualBackingAccess::recovery_bytes(*output_backing) != 0u) {
    return 3;
  }
  const std::shared_ptr<detail::DeviceState> &device_state =
      detail::DeviceAccess::state(*opened);
  detail::AccelDeviceState *const native =
      device_state == nullptr ? nullptr : detail::accel_device(*device_state);
  if (native == nullptr) {
    return 4;
  }
  const std::uint64_t submissions_before =
      rund::node::accel::ReadRuntimeStats(native->pick)
          .run.work.command_submit_count;
  if (!rund::node::accel::detail::InjectNativeDeviceLostOnce(native->pick)) {
    return 5;
  }

  const Status lost = prepared->run();
  const Stats unknown = prepared->stats();
  const BackingFacts after_unknown = output_backing->facts();
  const std::uint64_t submissions_after_unknown =
      rund::node::accel::ReadRuntimeStats(native->pick)
          .run.work.command_submit_count;
  const std::uint64_t unknown_version =
      detail::VirtualBackingAccess::version(*output_backing);
  const std::uint64_t unknown_recovery =
      detail::VirtualBackingAccess::recovery_bytes(*output_backing);

  const Status retry = prepared->run();
  const Stats after_retry = prepared->stats();
  const BackingFacts retried = output_backing->facts();
  const std::uint64_t submissions_after_retry =
      rund::node::accel::ReadRuntimeStats(native->pick)
          .run.work.command_submit_count;
  const bool exact =
      lost.reason() == Reason::DeviceLost &&
      retry.reason() == Reason::DeviceLost && unknown.command_submits == 1u &&
      unknown.dispatches == 1u && unknown.final_dispatches == 1u &&
      unknown.command_inflight_peak == 1u &&
      unknown.pipeline.residency.window_handoff_count == 1u &&
      unknown.pipeline.residency.window_queue_call_count == 1u &&
      unknown.pipeline.residency.page_out_count == 0u &&
      unknown.pipeline.residency.backing_write_bytes == 0u &&
      unknown.pipeline.residency.failed_page == 0u &&
      submissions_after_unknown == submissions_before + 1u &&
      submissions_after_retry == submissions_after_unknown &&
      after_unknown.write_count == published.write_count &&
      after_unknown.write_bytes == published.write_bytes &&
      retried.write_count == after_unknown.write_count &&
      retried.write_bytes == after_unknown.write_bytes &&
      unknown_version == version && unknown_recovery == LogicalBytes &&
      detail::VirtualBackingAccess::version(*output_backing) == version &&
      detail::VirtualBackingAccess::recovery_bytes(*output_backing) ==
          LogicalBytes &&
      after_retry == unknown;
  if (!exact) {
    std::fprintf(
        stderr,
        "DeviceVsm Unknown backend=%u status=%u/%u native=%llu/%llu/%llu "
        "submit=%llu dispatch=%llu final=%llu handoff=%llu queue=%llu "
        "out=%llu/%llu failed=%llu write=%llu/%llu version=%llu/%llu "
        "recovery=%llu\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(lost.reason()),
        static_cast<unsigned>(retry.reason()),
        static_cast<unsigned long long>(submissions_before),
        static_cast<unsigned long long>(submissions_after_unknown),
        static_cast<unsigned long long>(submissions_after_retry),
        static_cast<unsigned long long>(unknown.command_submits),
        static_cast<unsigned long long>(unknown.dispatches),
        static_cast<unsigned long long>(unknown.final_dispatches),
        static_cast<unsigned long long>(
            unknown.pipeline.residency.window_handoff_count),
        static_cast<unsigned long long>(
            unknown.pipeline.residency.window_queue_call_count),
        static_cast<unsigned long long>(
            unknown.pipeline.residency.page_out_count),
        static_cast<unsigned long long>(
            unknown.pipeline.residency.backing_write_bytes),
        static_cast<unsigned long long>(unknown.pipeline.residency.failed_page),
        static_cast<unsigned long long>(after_unknown.write_count),
        static_cast<unsigned long long>(retried.write_count),
        static_cast<unsigned long long>(version),
        static_cast<unsigned long long>(unknown_version),
        static_cast<unsigned long long>(unknown_recovery));
    return 6;
  }
  return 0;
#endif
}

} // namespace rund_node_test_virtual::product
