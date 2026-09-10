#include "local.hpp"

#include "src/compute/virtual/run/cache.hpp"
#include "src/compute/virtual/run/sliding/internal.hpp"
#include "src/compute/virtual/state.hpp"

#if defined(__APPLE__) && (defined(RUND_NODE_TEST_BACKEND_METAL) ||            \
                           defined(RUND_NODE_HAVE_METAL_SDK))
#include "src/accel/metal/kernel.hpp"
#endif

#include <cstdio>
#include <memory>

namespace rund_node_test_virtual::product::failure {
namespace {

#if defined(__APPLE__) && (defined(RUND_NODE_TEST_BACKEND_METAL) ||            \
                           defined(RUND_NODE_HAVE_METAL_SDK))
template <typename Pipeline>
[[nodiscard]] bool inspect_metal_persistent(
    Pipeline &pipeline,
    rund::node::accel::detail::MetalPersistentResidencySlidingDiagnostics
        &diagnostics) noexcept {
  const auto &state =
      rund::compute::detail::VirtualPipelineAccess::state(pipeline);
  std::shared_ptr<void> lowering{};
  if (state != nullptr && state->sliding_product_cache != nullptr &&
      state->sliding_product_cache->run != nullptr) {
    lowering = state->sliding_product_cache->run->persistent_preparation.backend
                   .lowering;
  }
  return lowering != nullptr &&
         rund::node::accel::detail::InspectMetalPersistentResidencySliding(
             lowering, diagnostics);
}
#endif

} // namespace

int CheckGenericBackingFailure(FailureFixture &fixture) {
  using namespace rund::compute;
  auto &input_backing = *fixture.input_backing;
  auto &prepared = *fixture.prepared;

  input_backing.fail_next_read(Reason::BackendFailed);
  const Status read_failed = prepared.run();
  const BackingFacts after_read_failure = input_backing.facts();
  const Stats read_failed_stats = prepared.stats();
#if defined(__APPLE__) && (defined(RUND_NODE_TEST_BACKEND_METAL) ||            \
                           defined(RUND_NODE_HAVE_METAL_SDK))
  bool metal_known_wait = true;
  if (fixture.backend == Backend::Metal) {
    rund::node::accel::detail::MetalPersistentResidencySlidingDiagnostics
        diagnostics{};
    const bool inspected = inspect_metal_persistent(prepared, diagnostics);
    metal_known_wait =
        inspected && diagnostics.last_wait.signaled &&
        diagnostics.last_wait.retired && !diagnostics.last_wait.admission &&
        diagnostics.last_wait.accepted && diagnostics.last_wait.reason == 0u &&
        diagnostics.last_wait.expected_descriptor_generation ==
            diagnostics.last_wait.observed_descriptor_generation &&
        diagnostics.first_failure.stage ==
            rund::node::accel::detail::
                MetalPersistentResidencySlidingDiagnosticStage::Unknown &&
        diagnostics.first_failure.predicate ==
            rund::node::accel::detail::
                MetalPersistentResidencySlidingDiagnosticPredicate::None;
    std::fprintf(
        stderr,
        "virtual read failure metal wait inspect=%u signal=%u retired=%u "
        "admission=%u accepted=%u reason=%u desc=%llu/%llu first=%u/%u "
        "\n",
        static_cast<unsigned>(inspected),
        static_cast<unsigned>(diagnostics.last_wait.signaled),
        static_cast<unsigned>(diagnostics.last_wait.retired),
        static_cast<unsigned>(diagnostics.last_wait.admission),
        static_cast<unsigned>(diagnostics.last_wait.accepted),
        diagnostics.last_wait.reason,
        static_cast<unsigned long long>(
            diagnostics.last_wait.expected_descriptor_generation),
        static_cast<unsigned long long>(
            diagnostics.last_wait.observed_descriptor_generation),
        static_cast<unsigned>(diagnostics.first_failure.stage),
        static_cast<unsigned>(diagnostics.first_failure.predicate));
  }
#endif
  const Status read_retry = prepared.run();
  const Stats read_retry_stats = prepared.stats();
  if (read_failed.reason() != Reason::BackendFailed ||
      after_read_failure.read_failure_count != 1u ||
      after_read_failure.read_count != 0u ||
      read_failed_stats.pipeline.residency.failed_page != 0u || !read_retry
#if defined(__APPLE__) && (defined(RUND_NODE_TEST_BACKEND_METAL) ||            \
                           defined(RUND_NODE_HAVE_METAL_SDK))
      || !metal_known_wait
#endif
  ) {
#if !defined(RUND_NODE_TEST_BACKEND_CPU)
    if (!read_retry) {
      rund::compute::detail::sliding_product_detail::ServiceFault
          service_fault{};
      const auto &retry_state =
          rund::compute::detail::VirtualPipelineAccess::state(prepared);
      if (retry_state != nullptr &&
          retry_state->sliding_product_cache != nullptr &&
          retry_state->sliding_product_cache->run != nullptr) {
        service_fault = rund::compute::detail::sliding_product_detail::
            snapshot_service_fault(*retry_state->sliding_product_cache->run);
      }
      std::fprintf(stderr, "virtual read retry service=%u/%u/%llu/%u/%llu\n",
                   static_cast<unsigned>(service_fault.stage),
                   service_fault.code,
                   static_cast<unsigned long long>(service_fault.coordinate),
                   service_fault.reason,
                   static_cast<unsigned long long>(service_fault.key));
    }
#endif
#if defined(__APPLE__) && (defined(RUND_NODE_TEST_BACKEND_METAL) ||            \
                           defined(RUND_NODE_HAVE_METAL_SDK))
    if (fixture.backend == Backend::Metal && !read_retry) {
      rund::node::accel::detail::MetalPersistentResidencySlidingDiagnostics
          diagnostics{};
      const bool inspected = inspect_metal_persistent(prepared, diagnostics);
      std::fprintf(
          stderr,
          "virtual read retry metal diag inspect=%u first=%u/%llu/%u/%llu "
          "wait=%llu/%u/%u/%u/%u/%u desc=%llu/%llu control=%llu/%llu\n",
          static_cast<unsigned>(inspected),
          static_cast<unsigned>(diagnostics.first_failure.stage),
          static_cast<unsigned long long>(diagnostics.first_failure.stage_key),
          static_cast<unsigned>(diagnostics.first_failure.predicate),
          static_cast<unsigned long long>(
              diagnostics.first_failure.predicate_key),
          static_cast<unsigned long long>(diagnostics.last_wait.coordinate),
          static_cast<unsigned>(diagnostics.last_wait.signaled),
          static_cast<unsigned>(diagnostics.last_wait.retired),
          static_cast<unsigned>(diagnostics.last_wait.admission),
          static_cast<unsigned>(diagnostics.last_wait.accepted),
          diagnostics.last_wait.reason,
          static_cast<unsigned long long>(
              diagnostics.last_wait.expected_descriptor_generation),
          static_cast<unsigned long long>(
              diagnostics.last_wait.observed_descriptor_generation),
          static_cast<unsigned long long>(
              diagnostics.last_wait.expected_control_generation),
          static_cast<unsigned long long>(
              diagnostics.last_wait.observed_control_generation));
    }
#endif
    std::fprintf(
        stderr,
        "virtual read failure backend=%u reason=%u failures=%llu reads=%llu "
        "failed_page=%llu retry=%u retry_page=%llu retry_epochs=%llu\n",
        static_cast<unsigned>(fixture.backend),
        static_cast<unsigned>(read_failed.reason()),
        static_cast<unsigned long long>(after_read_failure.read_failure_count),
        static_cast<unsigned long long>(after_read_failure.read_count),
        static_cast<unsigned long long>(
            read_failed_stats.pipeline.residency.failed_page),
        static_cast<unsigned>(read_retry.reason()),
        static_cast<unsigned long long>(
            read_retry_stats.pipeline.residency.failed_page),
        static_cast<unsigned long long>(
            read_retry_stats.pipeline.residency.epoch_count));
    return 5;
  }

  if (fixture.backend != Backend::Cpu) {
    rund::compute::detail::inject_residency_retention_failure_once();
    const Status retention_failed = prepared.run();
    const bool retention_not_selected =
        rund::compute::detail::cancel_residency_retention_failure_injection();
    const Status retention_retry = prepared.run();
    const bool exact =
        retention_not_selected
            ? static_cast<bool>(retention_failed)
            : retention_failed.reason() == Reason::PipelineInvalid;
    if (!exact || !retention_retry) {
#if defined(__APPLE__) && (defined(RUND_NODE_TEST_BACKEND_METAL) ||            \
                           defined(RUND_NODE_HAVE_METAL_SDK))
      if (fixture.backend == Backend::Metal && !retention_retry) {
        const auto &retry_state =
            detail::VirtualPipelineAccess::state(prepared);
        std::shared_ptr<void> lowering{};
        if (retry_state != nullptr &&
            retry_state->sliding_product_cache != nullptr &&
            retry_state->sliding_product_cache->run != nullptr) {
          lowering = retry_state->sliding_product_cache->run
                         ->persistent_preparation.backend.lowering;
        }
        rund::node::accel::detail::MetalPersistentResidencySlidingDiagnostics
            diagnostics{};
        const bool inspected =
            lowering != nullptr &&
            rund::node::accel::detail::InspectMetalPersistentResidencySliding(
                lowering, diagnostics);
        const bool trace_sane =
            !inspected ||
            (diagnostics.first_failure.stage !=
                 rund::node::accel::detail::
                     MetalPersistentResidencySlidingDiagnosticStage::Unknown &&
             diagnostics.first_failure.predicate !=
                 rund::node::accel::detail::
                     MetalPersistentResidencySlidingDiagnosticPredicate::None &&
             (diagnostics.last_wait.stage ==
                  rund::node::accel::detail::
                      MetalPersistentResidencySlidingDiagnosticStage::Unknown ||
              diagnostics.last_wait.predicate !=
                  rund::node::accel::detail::
                      MetalPersistentResidencySlidingDiagnosticPredicate::
                          None));
        std::fprintf(
            stderr,
            "virtual retention metal diag inspect=%u first=%u/%llu/%u/%llu "
            "wait=%llu/%u/%u/%u/%u/%u desc=%llu/%llu control=%llu/%llu "
            "trace=%u\n",
            static_cast<unsigned>(inspected),
            static_cast<unsigned>(diagnostics.first_failure.stage),
            static_cast<unsigned long long>(
                diagnostics.first_failure.stage_key),
            static_cast<unsigned>(diagnostics.first_failure.predicate),
            static_cast<unsigned long long>(
                diagnostics.first_failure.predicate_key),
            static_cast<unsigned long long>(diagnostics.last_wait.coordinate),
            static_cast<unsigned>(diagnostics.last_wait.signaled),
            static_cast<unsigned>(diagnostics.last_wait.retired),
            static_cast<unsigned>(diagnostics.last_wait.admission),
            static_cast<unsigned>(diagnostics.last_wait.accepted),
            diagnostics.last_wait.reason,
            static_cast<unsigned long long>(
                diagnostics.last_wait.expected_descriptor_generation),
            static_cast<unsigned long long>(
                diagnostics.last_wait.observed_descriptor_generation),
            static_cast<unsigned long long>(
                diagnostics.last_wait.expected_control_generation),
            static_cast<unsigned long long>(
                diagnostics.last_wait.observed_control_generation),
            static_cast<unsigned>(trace_sane));
      }
#endif
      std::fprintf(stderr,
                   "virtual retention failure backend=%u reason=%u "
                   "selected=%u retry=%u\n",
                   static_cast<unsigned>(fixture.backend),
                   static_cast<unsigned>(retention_failed.reason()),
                   static_cast<unsigned>(!retention_not_selected),
                   static_cast<unsigned>(retention_retry.reason()));
      return 9;
    }
  }
  return 0;
}

} // namespace rund_node_test_virtual::product::failure
