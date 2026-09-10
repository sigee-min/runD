#include "local.hpp"

#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)
#include "src/accel/kernel/fault.hpp"
#include "src/compute/backend.hpp"
#include "src/compute/device/state.hpp"
#include "src/compute/virtual/state.hpp"

#include "contract/target/selection.hpp"
#endif

#include <array>
#include <cstdio>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace rund_node_test_virtual::product::graph_resident::fixture_detail {
namespace {

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

using Owner =
    rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner;

[[nodiscard]] std::shared_ptr<rund::compute::VirtualBacking>
staged_backing(const std::span<const std::uint64_t> values,
               const std::shared_ptr<MemoryVirtualBackingCohort> &cohort) {
  auto backing = std::make_shared<MemoryVirtualBacking>(
      values.size() * sizeof(std::uint64_t),
      FrameElements * sizeof(std::uint64_t),
      rund::compute::VirtualBackingTier::Host, cohort);
  return backing->seed(std::as_bytes(values))
             ? std::static_pointer_cast<rund::compute::VirtualBacking>(
                   std::move(backing))
             : std::shared_ptr<rund::compute::VirtualBacking>{};
}

[[nodiscard]] Preparation
prepare_staged_case(const rund::compute::Device &device,
                    const Variant variant) {
  using namespace rund::compute;
  auto program = build_program(device, variant);
  if (!program || !validate_program(*program, variant)) {
    return {.reason = 11};
  }

  std::array<std::vector<std::uint64_t>, InputCount> values{};
  std::vector<std::uint64_t> expected(ElementCount);
  for (std::size_t index = 0u; index < ElementCount; ++index) {
    for (std::size_t input = 0u; input < InputCount; ++input) {
      values[input].push_back(Workload::input_value(input, index));
    }
    expected[index] = Workload::expected_value(index, variant);
  }

  std::array<std::shared_ptr<VirtualBacking>, InputCount> inputs{};
  const std::shared_ptr<MemoryVirtualBackingCohort> cohort =
      make_memory_backing_cohort();
  if (cohort == nullptr) {
    return {.reason = 12};
  }
  for (std::size_t index = 0u; index < InputCount; ++index) {
    inputs[index] = staged_backing(std::span{values[index]}, cohort);
    if (inputs[index] == nullptr) {
      return {.reason = 12};
    }
  }
  auto output = std::make_shared<MemoryVirtualBacking>(
      ElementCount * sizeof(std::uint64_t),
      FrameElements * sizeof(std::uint64_t));

  auto first = virtual_buffer<std::uint64_t>(ElementCount, inputs[0u]);
  auto second = virtual_buffer<std::uint64_t>(ElementCount, inputs[1u]);
  auto third = virtual_buffer<std::uint64_t>(ElementCount, inputs[2u]);
  auto result = virtual_buffer<std::uint64_t>(ElementCount, output);
  if (!first || !second || !third || !result) {
    return {.reason = 13};
  }
  auto prepared = virtual_pipeline(*program, *first, *second, *third, *result,
                                   ResidencyConfig{});
  if (!prepared || prepared.value().plan().residency.frame_capacity != 2u) {
    return {.reason = 14};
  }
  Pipeline pipeline = std::move(prepared).value();
  auto state = detail::VirtualPipelineAccess::state(pipeline);
  return {.value = std::make_unique<Case>(Case{
              .variant = variant,
              .first_values = std::move(values[0u]),
              .second_values = std::move(values[1u]),
              .third_values = std::move(values[2u]),
              .expected = std::move(expected),
              .inputs = std::move(inputs),
              .output = std::move(output),
              .pipeline = std::move(pipeline),
              .state = std::move(state),
          })};
}

[[nodiscard]] rund::compute::Result<Program>
build_dynamic_program(const rund::compute::Device &device) {
  return build_program(device, Variant::Ordinary);
}

[[nodiscard]] Preparation
prepare_dynamic_case(const rund::compute::Device &device) {
  using namespace rund::compute;
  auto program = build_dynamic_program(device);
  if (!program || !validate_program(*program, Variant::Ordinary)) {
    return {.reason = 15};
  }

  std::array<std::vector<std::uint64_t>, InputCount> values{};
  constexpr std::size_t DynamicPageCount = 6u;
  constexpr std::size_t DynamicElementCount =
      DynamicPageCount * FrameElements - TailElements;
  std::vector<std::uint64_t> expected(DynamicElementCount);
  for (std::size_t index = 0u; index < DynamicElementCount; ++index) {
    for (std::size_t input = 0u; input < InputCount; ++input) {
      values[input].push_back(Workload::input_value(input, index));
    }
    expected[index] = Workload::expected_value(index, Variant::Ordinary);
  }

  std::array<std::shared_ptr<VirtualBacking>, InputCount> inputs{};
  const std::shared_ptr<MemoryVirtualBackingCohort> cohort =
      make_memory_backing_cohort();
  if (cohort == nullptr) {
    return {.reason = 16};
  }
  for (std::size_t index = 0u; index < InputCount; ++index) {
    inputs[index] = staged_backing(std::span{values[index]}, cohort);
    if (inputs[index] == nullptr) {
      return {.reason = 16};
    }
  }
  auto output = std::make_shared<MemoryVirtualBacking>(
      DynamicElementCount * sizeof(std::uint64_t),
      FrameElements * sizeof(std::uint64_t));
  auto first = virtual_buffer<std::uint64_t>(DynamicElementCount, inputs[0u]);
  auto second = virtual_buffer<std::uint64_t>(DynamicElementCount, inputs[1u]);
  auto third = virtual_buffer<std::uint64_t>(DynamicElementCount, inputs[2u]);
  auto result = virtual_buffer<std::uint64_t>(DynamicElementCount, output);
  if (!first || !second || !third || !result) {
    return {.reason = 17};
  }
  auto prepared = virtual_pipeline(*program, *first, *second, *third, *result,
                                   ResidencyConfig{});
  if (!prepared || prepared.value().plan().residency.frame_capacity != 2u) {
    return {.reason = 18};
  }
  Pipeline pipeline = std::move(prepared).value();
  auto state = detail::VirtualPipelineAccess::state(pipeline);
  return {.value = std::make_unique<Case>(Case{
              .variant = Variant::Ordinary,
              .first_values = std::move(values[0u]),
              .second_values = std::move(values[1u]),
              .third_values = std::move(values[2u]),
              .expected = std::move(expected),
              .inputs = std::move(inputs),
              .output = std::move(output),
              .pipeline = std::move(pipeline),
              .state = std::move(state),
          })};
}

#endif

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

[[nodiscard]] bool run_dynamic_probe(const rund::compute::Backend backend,
                                     rund::compute::Device &device) {
  auto prepared = prepare_dynamic_case(device);
  if (!prepared.value) {
    std::fprintf(stderr, "GraphResident Q6 prepare reason=%d\n",
                 prepared.reason);
    return false;
  }
  Case &test_case = *prepared.value;
  Observation observation{};
  if (!run_once(test_case, observation, device, backend)) {
    return false;
  }
  return validate_dynamic_case(test_case, backend, observation);
}

#endif

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

[[nodiscard]] std::shared_ptr<Owner> retained(const Case &test_case) noexcept {
  return test_case.state == nullptr
             ? std::shared_ptr<Owner>{}
             : std::static_pointer_cast<Owner>(
                   test_case.state->device_vsm_product_cache);
}

#endif

} // namespace

bool run_staged_probe(const rund::compute::Backend backend,
                      const Variant variant) {
#if defined(RUND_NODE_TEST_BACKEND_CPU)
  (void)backend;
  (void)variant;
  return true;
#else
  auto opened =
      rund::compute::open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    return false;
  }
  rund::compute::Device &device = *opened;
  auto prepared = prepare_staged_case(device, variant);
  if (!prepared.value) {
    return false;
  }
  Case &test_case = *prepared.value;
  std::array<Observation, RunCount> observations{};
  for (Observation &observation : observations) {
    if (!run_once(test_case, observation, device, backend)) {
      return false;
    }
  }
  if (!validate_case(test_case, backend, observations)) {
    return false;
  }
  if (variant == Variant::Ordinary && !run_dynamic_probe(backend, device)) {
    return false;
  }

  const std::shared_ptr<Owner> before = retained(test_case);
  if (before == nullptr || before->evidence == nullptr ||
      test_case.state == nullptr) {
    return false;
  }
  const std::uint64_t version =
      rund::compute::detail::VirtualBackingAccess::version(*test_case.output);
  const auto &device_state = rund::compute::detail::DeviceAccess::state(device);
  rund::compute::detail::AccelDeviceState *const native =
      device_state == nullptr
          ? nullptr
          : rund::compute::detail::accel_device(*device_state);
  if (native == nullptr ||
      !rund::node::accel::detail::InjectNativeDeviceLostOnce(native->pick)) {
    return false;
  }
  Observation loss{};
  Observation retry{};
  if (!run_once(test_case, loss, device, backend) ||
      !run_once(test_case, retry, device, backend)) {
    return false;
  }
  const std::shared_ptr<Owner> after = retained(test_case);
  const std::uint64_t after_version =
      rund::compute::detail::VirtualBackingAccess::version(*test_case.output);
  const bool loss_native =
      loss.route_kind == RouteKind::Rejected &&
      loss.accepted_owner_mask == 0u && loss.accepted_owner_count == 0u &&
      loss.native_submit_count == 1u && loss.dispatch_count == 1u &&
      loss.epoch_submit_count == 0u && loss.final_count == 1u &&
      loss.native_may_write && !loss.host_service;
  const bool retry_idle =
      retry.native_submit_count == loss.native_submit_count &&
      retry.dispatch_count == loss.dispatch_count &&
      retry.final_count == loss.final_count &&
      retry.epoch_submit_count == loss.epoch_submit_count &&
      !retry.host_service && retry.native_may_write &&
      retry.after == loss.after;
  const bool loss_timing =
      loss.after.kernel_ns == loss.native_kernel_ns &&
      loss.after.kernel_samples == loss.native_kernel_samples &&
      loss.native_kernel_ns == 0u && loss.native_kernel_samples == 0u;
  const bool loss_wait =
      loss.after.submit_wait_ns == loss.native_submit_wait_ns;
  const bool retry_timing =
      retry.after.kernel_ns == loss.after.kernel_ns &&
      retry.after.kernel_samples == loss.after.kernel_samples &&
      retry.after.submit_wait_ns == loss.after.submit_wait_ns;
  return !loss.status &&
         loss.status.reason() == rund::compute::Reason::DeviceLost &&
         !retry.status &&
         retry.status.reason() == rund::compute::Reason::DeviceLost &&
         loss_native && loss_timing && loss_wait && retry_timing &&
         retry_idle && loss.quarantined && retry.quarantined &&
         loss.version_before == version && loss.version_after == version &&
         retry.version_before == version && retry.version_after == version &&
         after_version == version && after != nullptr &&
         after->evidence != nullptr && after->evidence->quarantined &&
         after->evidence->backing_publication_count == 0u;
#endif
}

} // namespace rund_node_test_virtual::product::graph_resident::fixture_detail
