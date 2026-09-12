#include "../../../target/selection.hpp"
#include "graph_forecast_window/internal.hpp"
#include "graph_forecast_window/worker.hpp"
#include "local.hpp"
#include "src/compute/device/residency/pool.hpp"
#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/state.hpp"
#include <algorithm>
#include <cstdio>

namespace rund_node_test_virtual::product {
namespace {
using namespace rund::compute;
using namespace graph_forecast_window;
template <class Pipeline>
int check_runs(Pipeline &pipeline, const std::shared_ptr<Control> &control,
               const std::span<const std::shared_ptr<Backing>> backings,
               const std::shared_ptr<Backing> &output, const Backend backend,
               const std::uint64_t adjustment) {
  const auto state = detail::VirtualPipelineAccess::state(pipeline);
  if (state == nullptr || state->graph_pipelines.size() != Stages * 2u)
    return 5;
  auto &pool = *state->pipeline->residency_pool;
  const std::uint64_t host_bytes = pool.host_bytes;
  const std::uint64_t storage_bytes = pool.host_storage_bytes;
  for (unsigned run = 0u; run < 7u; ++run) {
    const bool failure = run == 2u;
    control->reset(failure);
    for (auto &backing : backings)
      if (!backing->invalidate())
        return 6;
    const auto version = detail::VirtualBackingAccess::version(*output);
    const auto previous = output->values;
    const Status status = run_on_worker(pipeline);
    const Stats stats = pipeline.stats();
    if (failure) {
      if (status || status.reason() != Reason::BackendFailed ||
          output->values != previous ||
          detail::VirtualBackingAccess::version(*output) != version)
        return 7;
    } else {
      const bool exact =
          std::all_of(output->values.begin(), output->values.end(),
                      [index = std::size_t{0u},
                       adjustment](const std::uint64_t value) mutable {
                        return value == expected(index++) - adjustment;
                      });
      if (!status || !exact ||
          detail::VirtualBackingAccess::version(*output) != version + 1u ||
          stats.command_submits != Stages * 3u ||
          stats.pipeline.residency.backing_read_bytes !=
              backings.size() * Elements * sizeof(std::uint64_t) ||
          stats.pipeline.residency.backing_write_bytes !=
              Elements * sizeof(std::uint64_t)) {
        std::fprintf(stderr,
                     "forecast window backend=%u run=%u reason=%.*s exact=%u "
                     "submits=%llu reads=%llu writes=%llu\n",
                     static_cast<unsigned>(backend), run,
                     static_cast<int>(status.error().size()),
                     status.error().data(), static_cast<unsigned>(exact),
                     static_cast<unsigned long long>(stats.command_submits),
                     static_cast<unsigned long long>(
                         stats.pipeline.residency.backing_read_bytes),
                     static_cast<unsigned long long>(
                         stats.pipeline.residency.backing_write_bytes));
        return 8;
      }
    }
    if (!control->complete() || state->device_vsm_product_cache != nullptr ||
        pool.host_bytes != host_bytes ||
        pool.host_storage_bytes != storage_bytes ||
        !pool.prefetch[0u].quiescent() || !pool.prefetch[1u].quiescent()) {
      std::lock_guard lock{control->gate};
      std::fprintf(
          stderr,
          "forecast ownership inputs=%zu run=%u peak=%zu active=%zu refill=%u "
          "timeout=%u cache=%u host=%llu/%llu storage=%llu/%llu quiet=%u/%u\n",
          backings.size(), run, control->peak, control->active,
          static_cast<unsigned>(control->refilled_while_slow),
          static_cast<unsigned>(control->timeout),
          static_cast<unsigned>(state->device_vsm_product_cache != nullptr),
          static_cast<unsigned long long>(pool.host_bytes),
          static_cast<unsigned long long>(host_bytes),
          static_cast<unsigned long long>(pool.host_storage_bytes),
          static_cast<unsigned long long>(storage_bytes),
          static_cast<unsigned>(pool.prefetch[0u].quiescent()),
          static_cast<unsigned>(pool.prefetch[1u].quiescent()));
      return 9;
    }
  }
  return 0;
}
} // namespace

int CheckProductGraphForecastWindow(const rund::compute::Backend backend) {
  using namespace rund::compute;
  using namespace graph_forecast_window;

  auto device = open(rund::node::test_contract::target_for(backend));
  if (!device)
    return device.reason() == Reason::AdapterUnavailable ? 0 : 1;
  if (const int scratch = check_scratch(*device); scratch != 0)
    return 20 + scratch;
  if (backend == Backend::Cpu)
    return 0;
  auto program = build_program(*device);
  auto shared_program = build_shared_program(*device);
  if (!program || !shared_program)
    return 2;
  auto control = std::make_shared<Control>();
  std::array<std::shared_ptr<Backing>, Inputs> backings{};
  for (std::size_t input = 0u; input < Inputs; ++input)
    backings[input] = std::make_shared<Backing>(control, input);
  auto output = std::make_shared<Backing>(nullptr, Inputs);
  auto a = virtual_buffer<std::uint64_t>(Elements, backings[0u]);
  auto b = virtual_buffer<std::uint64_t>(Elements, backings[1u]);
  auto c = virtual_buffer<std::uint64_t>(Elements, backings[2u]);
  auto d = virtual_buffer<std::uint64_t>(Elements, backings[3u]);
  auto out = virtual_buffer<std::uint64_t>(Elements, output);
  if (!a || !b || !c || !d || !out)
    return 3;
  auto pipeline =
      virtual_pipeline(*program, *a, *b, *c, *d, *out, ResidencyConfig{});
  if (!pipeline)
    return 4;
  const int ordinary =
      check_runs(*pipeline, control, backings, output, backend, 0u);
  if (ordinary != 0)
    return ordinary;
  auto shared_pipeline =
      virtual_pipeline(*shared_program, *a, *b, *d, *out, ResidencyConfig{});
  if (!shared_pipeline)
    return 10;
  const std::array<std::shared_ptr<Backing>, 3u> shared{
      backings[0u], backings[1u], backings[3u]};
  // Two middle stages share b; the second must not block d while b is in
  // flight or overwrite b's Ready pin before its first consumer promotes.
  return check_runs(*shared_pipeline, control, shared, output, backend, Leaves);
}
} // namespace rund_node_test_virtual::product
