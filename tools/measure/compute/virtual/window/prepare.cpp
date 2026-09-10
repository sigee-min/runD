#include "local.hpp"

#include <new>
#include <span>
#include <utility>

namespace rund::measure::compute::virtual_window {
namespace {

[[nodiscard]] constexpr ::rund::compute::ResidencyConfig config() noexcept {
  return {.device_resident_bytes = ResidentBytes,
          .host_resident_bytes = ResidentBytes};
}

[[nodiscard]] bool valid_preparation(const Profile &profile,
                                     const Backend backend) noexcept {
  const auto &stats = profile.execution();
  if (!profile.memory().available() || profile.memory().backend != backend ||
      stats.backend != backend) {
    return false;
  }
  if (backend == Backend::Cpu) {
    return stats.pipeline.preparation_evidence ==
               ::rund::compute::PreparationEvidenceSource::NoNativeProducer &&
           stats.pipeline_compiles == 0u && stats.pipeline_cache_hits == 0u &&
           stats.pipeline_cache_evictions == 0u;
  }
  return backend == Backend::Metal &&
         stats.pipeline.preparation_evidence ==
             ::rund::compute::PreparationEvidenceSource::OwnerLocal &&
         stats.pipeline_compiles + stats.pipeline_cache_hits == 2u &&
         stats.pipeline_cache_evictions == 0u;
}

} // namespace

Prepared::Prepared(
    const Backend selected,
    std::shared_ptr<virtual_residency::MemoryBacking> input_owner,
    std::shared_ptr<virtual_residency::MemoryBacking> output_owner,
    Pipeline value) noexcept
    : backend(selected), input(std::move(input_owner)),
      output(std::move(output_owner)), pipeline(std::move(value)),
      plan(pipeline.plan()), observed(LogicalCapacity) {}

[[nodiscard]] std::unique_ptr<Prepared>
prepare(const Backend backend) noexcept {
  try {
    auto device = ::rund::compute::open(TargetFor(backend));
    if (!device) {
      return nullptr;
    }
    auto program =
        ::rund::compute::on(*device)
            .map<std::int32_t>("measure-virtual-window", PageElements,
                               [](auto value) { return (value + 5) * 3; })
            .compile();
    auto input = std::make_shared<virtual_residency::MemoryBacking>(
        LogicalCapacity * sizeof(std::int32_t));
    auto output = std::make_shared<virtual_residency::MemoryBacking>(
        LogicalCapacity * sizeof(std::int32_t));
    std::vector<std::int32_t> seeded(LogicalCapacity);
    seed(seeded);
    const auto seeded_status =
        input->write(0u, std::as_bytes(std::span{seeded}));
    auto virtual_input =
        ::rund::compute::virtual_buffer<std::int32_t>(LogicalCapacity, input);
    auto virtual_output =
        ::rund::compute::virtual_buffer<std::int32_t>(LogicalCapacity, output);
    auto pipeline =
        program && virtual_input && virtual_output
            ? ::rund::compute::virtual_pipeline(*program, *virtual_input,
                                                *virtual_output, config())
            : decltype(::rund::compute::virtual_pipeline(
                  *program, *virtual_input, *virtual_output,
                  config()))::fail(::rund::compute::Reason::PipelineInvalid);
    if (!seeded_status || !program || !virtual_input || !virtual_output ||
        !pipeline) {
      return nullptr;
    }
    auto prepared =
        std::make_unique<Prepared>(backend, std::move(input), std::move(output),
                                   std::move(pipeline).value());
    const auto preparation = prepared->pipeline.profile();
    if (!preparation || !valid_preparation(*preparation, backend)) {
      return nullptr;
    }
    return prepared;
  } catch (const std::bad_alloc &) {
    return nullptr;
  }
}

} // namespace rund::measure::compute::virtual_window
