#include "../model.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace rund_node_graph_services {
namespace {

template <class T>
[[nodiscard]] constexpr T memory_value(const std::int64_t raw) noexcept {
  if constexpr (std::is_integral_v<T>) {
    return static_cast<T>(raw);
  } else {
    return T::from_raw(static_cast<typename T::Raw>(raw));
  }
}

template <class T>
[[nodiscard]] bool check_memory_reuse(rund::compute::Device &device,
                                      const Backend backend,
                                      CanonicalReference &reference) {
  auto program =
      rund::compute::on(device)
          .map<T>("memory source", 4u,
                  [](auto value) {
                    if constexpr (rund::compute::detail::FixedValue<T>) {
                      return rund::compute::quantize<T>(value + value);
                    } else {
                      return value + value;
                    }
                  })
          .scan(rund::compute::Scan::InclusiveSum)
          .map("memory tail",
               [](auto value) {
                 if constexpr (rund::compute::detail::FixedValue<T>) {
                   return rund::compute::quantize<T>(value + value);
                 } else {
                   return value + value;
                 }
               })
          .scan(rund::compute::Scan::InclusiveSum)
          .compile();
  if (!program || !uses_backend(*program, backend)) {
    return false;
  }
  const Info &graph = program->graph();
  const std::uint64_t width = sizeof(T);
  std::vector<const rund::compute::graph::Resource *> internals;
  for (const auto &resource : graph.resources) {
    if (resource.visibility == Visibility::Internal) {
      internals.push_back(&resource);
    }
  }
  if (!ValidResourceGraph(graph) || graph.nodes.size() != 4u ||
      internals.size() != 3u || graph.memory.logical_bytes != 12u * width ||
      graph.memory.live_bytes != 8u * width ||
      graph.memory.physical_bytes != 256u + 4u * width ||
      graph.memory.allocation_count != 1u || internals[0u]->first_use != 0u ||
      internals[0u]->last_use != 1u || internals[1u]->first_use != 1u ||
      internals[1u]->last_use != 2u || internals[2u]->first_use != 2u ||
      internals[2u]->last_use != 3u ||
      internals[0u]->alias_group != internals[1u]->alias_group ||
      internals[1u]->alias_group != internals[2u]->alias_group ||
      internals[1u]->alias_offset_bytes != internals[2u]->alias_offset_bytes ||
      internals[2u]->source != internals[1u]->id) {
    return false;
  }
  const std::array<T, 4u> input{memory_value<T>(1), memory_value<T>(2),
                                memory_value<T>(3), memory_value<T>(4)};
  const std::array<T, 4u> expected{memory_value<T>(4), memory_value<T>(16),
                                   memory_value<T>(40), memory_value<T>(80)};
  auto job = program->resident(std::span<const T>{input});
  if (!job || !job->run() || !job->run()) {
    return false;
  }
  const auto warm = job->stats();
  if (warm.pipeline_compiles != 0u || warm.buffer_allocations != 0u ||
      warm.uploaded_bytes != 0u || warm.download_events != 0u ||
      warm.command_submits != (backend == Backend::Cpu ? 0u : 1u)) {
    return false;
  }
  auto output = job->read();
  const auto stats = job->stats();
  const ExecutionEvidence execution{.graph_hash = stats.graph_hash,
                                    .output_hash = stats.output_hash};
  return output &&
         *output == std::vector<T>{expected.begin(), expected.end()} &&
         stats.backend == backend && execution.graph_hash != 0u &&
         execution.output_hash != 0u &&
         MatchReference(reference, graph, execution);
}

} // namespace

bool CheckMemoryReuse(rund::compute::Device &device, const Backend backend,
                      std::array<CanonicalReference, 4u> &references) {
  return check_memory_reuse<std::uint32_t>(device, backend, references[0u]) &&
         check_memory_reuse<std::uint64_t>(device, backend, references[1u]) &&
         check_memory_reuse<rund::compute::Fixed<16, 16>>(device, backend,
                                                          references[2u]) &&
         check_memory_reuse<rund::compute::Fixed<20, 44>>(device, backend,
                                                          references[3u]);
}

} // namespace rund_node_graph_services
