#include "../../../target/selection.hpp"
#include "graph_forecast_window/internal.hpp"
#include "local.hpp"
#include "src/compute/device/residency/pool.hpp"
#include "src/compute/virtual/backing.hpp"
#include "src/compute/virtual/state.hpp"
#include <algorithm>
#include <cstdio>

namespace rund_node_test_virtual::product {
int CheckProductGraphForecastWindow(const rund::compute::Backend backend) {
  using namespace rund::compute;
  using namespace graph_forecast_window;
  if (backend == Backend::Cpu)
    return 0;
  auto device = open(rund::node::test_contract::target_for(backend));
  if (!device)
    return device.reason() == Reason::AdapterUnavailable ? 0 : 1;
  auto program = build_program(*device);
  if (!program)
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
  const auto state = detail::VirtualPipelineAccess::state(*pipeline);
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
    const Status status = pipeline->run();
    const Stats stats = pipeline->stats();
    if (failure) {
      if (status || status.reason() != Reason::BackendFailed ||
          output->values != previous ||
          detail::VirtualBackingAccess::version(*output) != version)
        return 7;
    } else {
      const bool exact = std::all_of(
          output->values.begin(), output->values.end(),
          [index = std::size_t{0u}](const std::uint64_t value) mutable {
            return value == expected(index++);
          });
      if (!status || !exact ||
          detail::VirtualBackingAccess::version(*output) != version + 1u ||
          stats.command_submits != Stages * 3u ||
          stats.pipeline.residency.backing_read_bytes !=
              Inputs * Elements * sizeof(std::uint64_t) ||
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
        !pool.prefetch[0u].quiescent() || !pool.prefetch[1u].quiescent())
      return 9;
  }
  return 0;
}
} // namespace rund_node_test_virtual::product
