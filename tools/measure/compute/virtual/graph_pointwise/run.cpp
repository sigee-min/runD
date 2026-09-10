#include "internal.hpp"
#include "run/local.hpp"

#include "../../suite/core.hpp"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rund::measure::compute::virtual_graph_pointwise {

bool Run(::rund::compute::Device &device,
         const ::rund::compute::DeviceInfo &info, Result &result) {
  using namespace run_detail;
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
  Case test_case{};
  const auto cold_begin = Clock::now();
  if (!prepare(device, test_case)) {
    return false;
  }
  result.graph_hi = test_case.graph_hi;
  result.graph_lo = test_case.graph_lo;
  std::uint64_t input_before = 0u;
  if (!input_digest(test_case, input_before) ||
      input_before != Spec::input_digest()) {
    return false;
  }
  result.input_digest_before = input_before;
  std::vector<std::uint64_t> output(Spec::ElementCount);
  const auto version_before = Access::version(*test_case.output);
  const auto cold_status = test_case.pipeline->run();
  const auto cold_route = observer.evidence();
  const bool cold_read = read_output(test_case.output, output);
  const auto cold_profile = test_case.pipeline->profile();
  const auto cold_stats = test_case.pipeline->stats();
  const auto cold_end = Clock::now();
  result.cold_us = micros(cold_end - cold_begin);
  result.cold = facts(test_case, backend, cold_status, cold_route,
                      version_before, Access::version(*test_case.output),
                      cold_read && cold_profile && expected_output(output),
                      output, cold_stats);
  if (!result.cold.ok) {
    return false;
  }

  observer.reset();
  const auto conditioning = test_case.pipeline->run();
  if (!conditioning) {
    record_status(result.warm, conditioning);
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
      backend == Backend::Cpu ||
      (conditioned_owner != nullptr &&
       reinterpret_cast<std::uintptr_t>(conditioned_owner.get()) ==
           result.cold.owner_identity &&
       reinterpret_cast<std::uintptr_t>(
           conditioned_owner->registration.get()) ==
           result.cold.control_identity);

  const auto warm_before = Access::version(*test_case.output);
  observer.reset();
  const auto begin_samples = test_case.pipeline->begin_samples();
  if (!begin_samples) {
    record_status(result.warm, begin_samples);
    return false;
  }
  ::rund::compute::Status warm_status = conditioning;
  bool warm_failed = false;
  for (std::size_t sample = 0u; sample < SampleCount; ++sample) {
    const auto begin = Clock::now();
    const auto status = test_case.pipeline->run();
    const auto end = Clock::now();
    result.timing.samples[sample] = micros(end - begin);
    if (!status) {
      warm_status = status;
      warm_failed = true;
      break;
    }
    warm_status = status;
    ++result.completed_samples;
  }
  if (warm_failed) {
    result.timing_complete = false;
    result.warm = facts(test_case, backend, warm_status, observer.evidence(),
                        warm_before, Access::version(*test_case.output), false,
                        output, test_case.pipeline->stats());
    result.ok = false;
    return false;
  }
  const auto end_samples = test_case.pipeline->end_samples();
  if (!end_samples) {
    record_status(result.warm, end_samples);
    return false;
  }
  const bool timing_ok =
      virtual_graph_residency::summarize_timing(result.timing);
  result.timing_complete = result.completed_samples == SampleCount && timing_ok;
  const auto warm_profile = test_case.pipeline->profile();
  const bool warm_read = read_output(test_case.output, output);
  const auto warm_route = observer.evidence();
  std::uint64_t input_after = 0u;
  const bool inputs_unchanged = input_digest(test_case, input_after);
  result.input_digest_after = inputs_unchanged ? input_after : 0u;
  const auto warm_stats = test_case.pipeline->stats();
  result.warm = facts(test_case, backend, warm_status, warm_route, warm_before,
                      Access::version(*test_case.output),
                      warm_read && warm_profile && expected_output(output),
                      output, warm_stats);
  result.owner_stable =
      owner_after_conditioning &&
      (backend == Backend::Cpu ||
       (result.cold.owner_identity != 0u &&
        result.cold.control_identity != 0u &&
        result.cold.owner_identity == result.warm.owner_identity &&
        result.cold.control_identity == result.warm.control_identity));
  result.ok =
      timing_ok && inputs_unchanged && result.warm.ok && Validate(result);
  return result.ok;
}

namespace {

void report_failure(const Backend backend, const char *status,
                    const std::uint32_t code, const std::string_view error) {
  std::printf("environment,%s,%s,%u,", Name(backend), status,
              static_cast<unsigned>(code));
  PrintCsv(error);
  std::fputs(",\"\",\"\",\"\"\n", stdout);
}

} // namespace

bool Measure(const Backend backend) {
  Result result{};
  result.backend = backend;
  bool ok = true;
  auto device = ::rund::compute::open(TargetFor(backend));
  if (!device) {
    result.device_code = static_cast<std::uint32_t>(device.code());
    result.device_error = std::string(device.error());
    report_failure(backend, "open_failed", result.device_code, device.error());
    ok = false;
  } else {
    const auto info = device->info();
    if (!info) {
      result.device_code = static_cast<std::uint32_t>(info.code());
      result.device_error = std::string(info.error());
      report_failure(backend, "info_failed", result.device_code, info.error());
      ok = false;
    } else {
      ok = ReportEnvironment(backend, *device, *info) && ok;
      ok = info->backend == backend && ok;
      result.device_valid = info->backend == backend;
      result.device = info->name;
      result.driver = info->driver;
      result.driver_details = info->driver_details;
      result.device_code = static_cast<std::uint32_t>(
          info->backend == backend ? ::rund::compute::Code::Ok
                                   : ::rund::compute::Code::Invalid);
      if (info->backend != backend) {
        result.device_error = "compute_device_info_backend_mismatch";
      }
      ::rund::measure::compute::PrintVirtualGraphPointwiseColumns();
      ok = Run(*device, *info, result) && ok;
      Report(result);
      return ok;
    }
  }
  ::rund::measure::compute::PrintVirtualGraphPointwiseColumns();
  Report(result);
  return false;
}

} // namespace rund::measure::compute::virtual_graph_pointwise
