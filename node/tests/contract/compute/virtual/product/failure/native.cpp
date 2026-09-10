#include "local.hpp"

#include "../golden.hpp"

#include "src/compute/backend.hpp"
#include "src/compute/device/state.hpp"
#include "src/compute/virtual/state.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)
#include "src/accel/kernel/fault.hpp"
#endif

#if defined(__APPLE__) && (defined(RUND_NODE_TEST_BACKEND_METAL) ||            \
                           defined(RUND_NODE_HAVE_METAL_SDK))
#include "src/accel/metal/kernel.hpp"
#endif

#include <array>
#include <cstdio>
#include <memory>

namespace rund_node_test_virtual::product::failure {

int CheckNativeTransferFailure(FailureFixture &fixture) {
#if !defined(RUND_NODE_TEST_BACKEND_CPU)
  using namespace rund::compute;
  if (fixture.backend == Backend::Metal) {
    auto &prepared = *fixture.prepared;
    auto &output_backing = *fixture.output_backing;
    const auto &prepared_state = detail::VirtualPipelineAccess::state(prepared);
    if (prepared_state == nullptr) {
      return 20;
    }
    const bool prior_device_vsm_required =
        prepared_state->geometry.device_vsm_required;
    // The native host-transfer faults below are explicit DeviceVsm fallback
    // contracts. Opt this block in after the ordinary default-route checks.
    prepared_state->geometry.device_vsm_required = true;
    const std::shared_ptr<detail::DeviceState> &device_state =
        detail::DeviceAccess::state(*fixture.device);
    detail::AccelDeviceState *const native =
        device_state == nullptr ? nullptr : detail::accel_device(*device_state);
    if (native == nullptr ||
        !rund::node::accel::detail::InjectNativeHostWriteUnavailableOnce(
            native->pick)) {
      return 20;
    }
    const Status input_fallback = prepared.run();
    const Stats input_fallback_stats = prepared.stats();
    const Status input_retry = prepared.run();
    const Stats input_retry_stats = prepared.stats();
    if (!input_fallback || input_fallback_stats.uploaded_bytes == 0u ||
        input_fallback_stats.transfer_submissions.host_to_device != 0u ||
        !input_retry || input_retry_stats.uploaded_bytes != 0u ||
        input_retry_stats.transfer_submissions.host_to_device != 0u) {
      std::fprintf(
          stderr,
          "virtual direct input fallback reason=%u uploaded=%llu submit=%llu "
          "retry=%u/%llu/%llu\n",
          static_cast<unsigned>(input_fallback.reason()),
          static_cast<unsigned long long>(input_fallback_stats.uploaded_bytes),
          static_cast<unsigned long long>(
              input_fallback_stats.transfer_submissions.host_to_device),
          static_cast<unsigned>(input_retry.reason()),
          static_cast<unsigned long long>(input_retry_stats.uploaded_bytes),
          static_cast<unsigned long long>(
              input_retry_stats.transfer_submissions.host_to_device));
      return 21;
    }
    const BackingFacts before_physical_fallback = output_backing.facts();
    if (!rund::node::accel::detail::InjectNativeHostReadUnavailableOnce(
            native->pick)) {
      return 16;
    }
    const Status physical_fallback = prepared.run();
    const Stats physical_fallback_stats = prepared.stats();
    const BackingFacts after_physical_fallback = output_backing.facts();
    const Status coherent_retry = prepared.run();
    const Stats coherent_retry_stats = prepared.stats();
    std::array<std::int32_t, LogicalElements> observed{};
    if (!physical_fallback ||
        // DeviceVsm owns one whole-run output Buffer. A denied coherent view
        // therefore downloads that exact logical extent once before the same
        // page-sliced backing publication; it does not expose a rolling bank.
        physical_fallback_stats.downloaded_bytes != LogicalBytes ||
        physical_fallback_stats.pipeline.residency.backing_write_bytes !=
            LogicalBytes ||
        after_physical_fallback.write_count -
                before_physical_fallback.write_count !=
            PageCount ||
        after_physical_fallback.write_bytes -
                before_physical_fallback.write_bytes !=
            LogicalBytes ||
        !coherent_retry || coherent_retry_stats.downloaded_bytes != 0u ||
        !output_backing.observe(std::as_writable_bytes(std::span{observed})) ||
        !GoldenMatches(observed)) {
      std::fprintf(
          stderr,
          "virtual direct physical fallback reason=%u downloaded=%llu "
          "backing=%llu writes=%llu bytes=%llu coherent=%u/%llu\n",
          static_cast<unsigned>(physical_fallback.reason()),
          static_cast<unsigned long long>(
              physical_fallback_stats.downloaded_bytes),
          static_cast<unsigned long long>(
              physical_fallback_stats.pipeline.residency.backing_write_bytes),
          static_cast<unsigned long long>(after_physical_fallback.write_count -
                                          before_physical_fallback.write_count),
          static_cast<unsigned long long>(after_physical_fallback.write_bytes -
                                          before_physical_fallback.write_bytes),
          static_cast<unsigned>(coherent_retry.reason()),
          static_cast<unsigned long long>(
              coherent_retry_stats.downloaded_bytes));
      return 17;
    }

    const BackingFacts before_download_failure = output_backing.facts();
    if (!rund::node::accel::detail::InjectNativeDownloadFailureOnce(
            native->pick)) {
      return 18;
    }
    const Status download_failed = prepared.run();
    const BackingFacts after_download_failure = output_backing.facts();
    const Status download_retry = prepared.run();
    if (download_failed.reason() != Reason::TransferInvalid ||
        after_download_failure.write_count !=
            before_download_failure.write_count ||
        after_download_failure.write_failure_count !=
            before_download_failure.write_failure_count ||
        !download_retry ||
        !output_backing.observe(std::as_writable_bytes(std::span{observed})) ||
        !GoldenMatches(observed)) {
      std::fprintf(
          stderr,
          "virtual direct download fallback reason=%u writes=%llu/%llu "
          "write_failures=%llu/%llu retry=%u\n",
          static_cast<unsigned>(download_failed.reason()),
          static_cast<unsigned long long>(after_download_failure.write_count),
          static_cast<unsigned long long>(before_download_failure.write_count),
          static_cast<unsigned long long>(
              after_download_failure.write_failure_count),
          static_cast<unsigned long long>(
              before_download_failure.write_failure_count),
          static_cast<unsigned>(download_retry.reason()));
      return 19;
    }
    prepared_state->geometry.device_vsm_required = prior_device_vsm_required;
  }
#else
  (void)fixture;
#endif
  return 0;
}

} // namespace rund_node_test_virtual::product::failure
