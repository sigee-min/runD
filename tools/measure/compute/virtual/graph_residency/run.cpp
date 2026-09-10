#include "run/local.hpp"

#include "../backing.hpp"

#include <utility>
#include <vector>

namespace rund::measure::compute::virtual_graph_residency {

using namespace run_detail;

bool Run(::rund::compute::Device &device,
         const ::rund::compute::DeviceInfo &info, Result &result) {
  const Backend backend = result.backend;
  if (backend == Backend::Unavailable || info.backend != backend) {
    result.device_code =
        static_cast<std::uint32_t>(::rund::compute::Code::Invalid);
    result.device_error = "compute_device_info_backend_mismatch";
    return false;
  }
  result.device_valid = true;
  result.device_code = static_cast<std::uint32_t>(::rund::compute::Code::Ok);
  result.device = info.name;
  result.driver = info.driver;
  result.driver_details = info.driver_details;
  virtual_residency::ProductRouteObserver observer{device};
  observer.reset();
  result.input_digest_before = Spec::input_digest();
  Case test_case{};
  const auto cold_begin = Clock::now();
  if (!prepare(device, backend, test_case)) {
    return false;
  }
  std::uint64_t input_before = 0u;
  if (!input_digest(test_case, input_before) ||
      input_before != Spec::input_digest()) {
    return false;
  }
  result.input_digest_before = input_before;
  std::vector<std::uint64_t> output(Spec::ElementCount);
  const auto version_before = Access::version(*test_case.output);
  virtual_residency::ProductRouteEvidence route{};
  ::rund::compute::Status cold_status =
      ::rund::compute::Status::fail(::rund::compute::Reason::CompletionInvalid);
  cold_status = test_case.pipeline->run();
  route = observer.evidence();
  const bool cold_read = read_output(test_case.output, output);
  const auto cold_profile = test_case.pipeline->profile();
  const auto cold_stats = test_case.pipeline->stats();
  const auto cold_end = Clock::now();
  result.cold_us = micros(cold_end - cold_begin);
  result.cold = facts(test_case, cold_status, route, version_before,
                      Access::version(*test_case.output),
                      cold_read && cold_profile && expected_output(output),
                      output, cold_stats);
  if (!result.cold.ok) {
    return false;
  }

  observer.reset();
  const auto conditioning = test_case.pipeline->run();
  if (!conditioning) {
    return false;
  }
  result.conditioning = observer.evidence();
  const auto conditioned_state =
      ::rund::compute::detail::VirtualPipelineAccess::state(
          *test_case.pipeline);
  const auto conditioned_owner =
      conditioned_state == nullptr
          ? std::shared_ptr<Owner>{}
          : std::static_pointer_cast<Owner>(
                conditioned_state->device_vsm_product_cache);
  const bool owner_after_conditioning =
      conditioned_owner != nullptr &&
      reinterpret_cast<std::uintptr_t>(conditioned_owner.get()) ==
          result.cold.owner_identity &&
      reinterpret_cast<std::uintptr_t>(conditioned_owner->registration.get()) ==
          result.cold.control_identity;
  const auto warm_before = Access::version(*test_case.output);
  observer.reset();
  ::rund::compute::Status warm_status = conditioning;
  for (std::size_t sample = 0u; sample < SampleCount; ++sample) {
    const std::uint64_t enter = nanos();
    const auto status = test_case.pipeline->run();
    const std::uint64_t run_end = nanos();
    if (!status) {
      return false;
    }
    const auto sample_state =
        ::rund::compute::detail::VirtualPipelineAccess::state(
            *test_case.pipeline);
    const auto sample_owner = sample_state == nullptr
                                  ? std::shared_ptr<Owner>{}
                                  : std::static_pointer_cast<Owner>(
                                        sample_state->device_vsm_product_cache);
    ::rund::node::accel::detail::DeviceVsmEvidence native{};
    if (sample_owner != nullptr && sample_owner->evidence != nullptr) {
      native = sample_owner->evidence->native;
    }
    result.timing.samples[sample] = phase_us(enter, run_end);
    if (phase_ready(status, sample_owner, native, enter, run_end,
                    result.cold.owner_identity, result.cold.control_identity)) {
      auto &phases = result.timing.phases;
      phases.pre.samples[sample] = phase_us(enter, native.submit_ns);
      phases.queue.samples[sample] =
          phase_us(native.submit_ns, native.callback_ns);
      phases.finish.samples[sample] =
          phase_us(native.callback_ns, native.final_ns);
      phases.post.samples[sample] = phase_us(native.final_ns, run_end);
      ++phases.pre.valid;
      ++phases.queue.valid;
      ++phases.finish.valid;
      ++phases.post.valid;
    }
    warm_status = status;
  }
  const bool timing_ok = summarize_timing(result.timing);
  const bool phase_ok =
      summarize_phases(result.timing.phases) || backend == Backend::Cpu;
  const auto warm_profile = test_case.pipeline->profile();
  const bool warm_read = read_output(test_case.output, output);
  const auto warm_route = observer.evidence();
  std::uint64_t input_after = 0u;
  const bool inputs_unchanged = input_digest(test_case, input_after);
  result.input_digest_after = inputs_unchanged ? input_after : 0u;
  const auto warm_stats = test_case.pipeline->stats();
  result.warm = facts(test_case, warm_status, warm_route, warm_before,
                      Access::version(*test_case.output),
                      warm_read && warm_profile && expected_output(output),
                      output, warm_stats);
  result.owner_stable =
      owner_after_conditioning && result.cold.owner_identity != 0u &&
      result.cold.control_identity != 0u &&
      result.cold.owner_identity == result.warm.owner_identity &&
      result.cold.control_identity == result.warm.control_identity;
  result.ok = timing_ok && phase_ok && inputs_unchanged && result.warm.ok &&
              Validate(result);
  return result.ok;
}

} // namespace rund::measure::compute::virtual_graph_residency
