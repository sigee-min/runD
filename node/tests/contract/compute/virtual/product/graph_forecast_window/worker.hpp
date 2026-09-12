#pragma once

#include <rund/compute.hpp>

#if defined(__APPLE__) || defined(__linux__)
#include <pthread.h>
#endif

namespace rund_node_test_virtual::product::graph_forecast_window {

// Only public run() runs on this stack. Preparation and the test fixture stay
// on the caller, so this catches execution scratch growth instead of test data.
template <class Pipeline>
[[nodiscard]] rund::compute::Status run_on_worker(Pipeline &pipeline) {
#if defined(__APPLE__) || defined(__linux__)
  struct Call final {
    Pipeline &pipeline;
    rund::compute::Status status{
        rund::compute::Status::fail(rund::compute::Reason::BackendFailed)};
  } call{pipeline};
  pthread_attr_t attributes{};
  if (pthread_attr_init(&attributes) != 0)
    return call.status;
  // One MiB is also used under instrumentation. Release probes additionally
  // cover 512 KiB, without turning an ABI-specific frame size into an API law.
  const int sized = pthread_attr_setstacksize(&attributes, 1024u * 1024u);
  pthread_t thread{};
  const int created = sized == 0
                          ? pthread_create(
                                &thread, &attributes,
                                [](void *opaque) -> void * {
                                  auto &entry = *static_cast<Call *>(opaque);
                                  entry.status = entry.pipeline.run();
                                  return nullptr;
                                },
                                &call)
                          : sized;
  (void)pthread_attr_destroy(&attributes);
  if (created != 0)
    return call.status;
  // Returning before join would let the worker access a dead Call.
  if (pthread_join(thread, nullptr) != 0)
    std::terminate();
  return call.status;
#else
  return pipeline.run();
#endif
}

} // namespace rund_node_test_virtual::product::graph_forecast_window
