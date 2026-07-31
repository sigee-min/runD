#include "local/model.hpp"

#include <rund/compute.hpp>
#include <rund/compute/math.hpp>

#include "../../../target/selection.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <vector>

namespace rund::node::test_contract::numeric {
namespace {

template <class T> [[nodiscard]] T Half() noexcept {
  if constexpr (std::same_as<T, rund::compute::Fixed<1, 31>>) {
    return T::from_raw(std::int32_t{1} << 30);
  } else {
    return T::from_raw(std::int64_t{1} << 62);
  }
}

template <class T> [[nodiscard]] T Quarter() noexcept {
  if constexpr (std::same_as<T, rund::compute::Fixed<1, 31>>) {
    return T::from_raw(std::int32_t{1} << 29);
  } else {
    return T::from_raw(std::int64_t{1} << 61);
  }
}

template <class Job>
[[nodiscard]] bool Warm(Job &job, const rund::compute::Backend backend,
                        const char *const operation) {
  const rund::compute::Status first = job.run();
  if (!first) {
    std::fprintf(stderr, "%s warm first backend=%u error=%.*s\n", operation,
                 static_cast<unsigned>(backend),
                 static_cast<int>(first.error().size()), first.error().data());
    return false;
  }
  const rund::compute::Status second = job.run();
  if (!second) {
    std::fprintf(stderr, "%s warm second backend=%u error=%.*s\n", operation,
                 static_cast<unsigned>(backend),
                 static_cast<int>(second.error().size()),
                 second.error().data());
    return false;
  }
  const rund::compute::Stats stats = job.stats();
  const bool stable = stats.pipeline_compiles == 0u &&
                      stats.buffer_allocations == 0u &&
                      stats.uploaded_bytes == 0u && stats.download_events == 0u;
  if (!stable) {
    std::fprintf(stderr,
                 "%s warm stats backend=%u pipelines=%llu allocations=%llu "
                 "uploaded=%llu downloads=%llu\n",
                 operation, static_cast<unsigned>(backend),
                 static_cast<unsigned long long>(stats.pipeline_compiles),
                 static_cast<unsigned long long>(stats.buffer_allocations),
                 static_cast<unsigned long long>(stats.uploaded_bytes),
                 static_cast<unsigned long long>(stats.download_events));
  }
  return stable;
}

template <class T>
[[nodiscard]] int CheckFactor(const std::array<T, 4u> &input) {
  using namespace rund::compute;
  const auto make = [](const Backend selected) {
    return on(rund::node::test_contract::target_for(selected))
        .template map<T>("factor-parity", 4u,
                         [](auto value) { return quantize<T>(value); })
        .template matrix<2u, 2u>()
        .lu()
        .compile();
  };
  auto cpu_program = make(Backend::Cpu);
  if (!cpu_program) {
    return 1;
  }
  auto cpu = cpu_program->resident(input);
  if (!cpu) {
    std::fprintf(stderr, "factor resident backend=%u error=%.*s\n",
                 static_cast<unsigned>(Backend::Cpu),
                 static_cast<int>(cpu.error().size()), cpu.error().data());
    return 2;
  }
  if (!Warm(*cpu, Backend::Cpu, "factor")) {
    return 2;
  }
  const Stats cpu_stats = cpu->stats();
  auto cpu_output = cpu->read_all();
  if (!cpu_output) {
    return 3;
  }
  for (const Backend backend :
       rund::node::test_contract::selected_accelerators()) {
    auto target_program = make(backend);
    if (!target_program) {
      return 1;
    }
    auto target = target_program->resident(input);
    if (!target) {
      std::fprintf(stderr, "factor resident backend=%u error=%.*s\n",
                   static_cast<unsigned>(backend),
                   static_cast<int>(target.error().size()),
                   target.error().data());
      return 2;
    }
    if (!Warm(*target, backend, "factor")) {
      return 2;
    }
    const Stats target_stats = target->stats();
    auto target_output = target->read_all();
    if (!target_output || *cpu_output != *target_output ||
        cpu_stats.graph_hash != target_stats.graph_hash ||
        cpu_stats.output_hash != target_stats.output_hash) {
      return 3;
    }
  }
  return 0;
}

template <class T>
[[nodiscard]] int CheckSolve(const std::array<T, 4u> &input) {
  using namespace rund::compute;
  const auto make = [](const Backend selected) {
    return on(rund::node::test_contract::target_for(selected))
        .template map<T>("solve-parity", 4u,
                         [](auto value) { return quantize<T>(value); })
        .template matrix<2u, 2u>()
        .lu()
        .template solve<2u>()
        .compile();
  };
  auto cpu_program = make(Backend::Cpu);
  if (!cpu_program) {
    return 1;
  }
  auto cpu = cpu_program->resident(input, input);
  if (!cpu) {
    std::fprintf(stderr, "solve resident backend=%u error=%.*s\n",
                 static_cast<unsigned>(Backend::Cpu),
                 static_cast<int>(cpu.error().size()), cpu.error().data());
    return 2;
  }
  if (!Warm(*cpu, Backend::Cpu, "solve")) {
    return 2;
  }
  const Stats cpu_stats = cpu->stats();
  auto cpu_output = cpu->read_all();
  if (!cpu_output) {
    return 3;
  }
  for (const Backend backend :
       rund::node::test_contract::selected_accelerators()) {
    auto target_program = make(backend);
    if (!target_program) {
      return 1;
    }
    auto target = target_program->resident(input, input);
    if (!target) {
      std::fprintf(stderr, "solve resident backend=%u error=%.*s\n",
                   static_cast<unsigned>(backend),
                   static_cast<int>(target.error().size()),
                   target.error().data());
      return 2;
    }
    if (!Warm(*target, backend, "solve")) {
      return 2;
    }
    const Stats target_stats = target->stats();
    auto target_output = target->read_all();
    if (!target_output || *cpu_output != *target_output ||
        cpu_stats.graph_hash != target_stats.graph_hash ||
        cpu_stats.output_hash != target_stats.output_hash) {
      return 3;
    }
  }
  return 0;
}

template <class T>
[[nodiscard]] int CheckSpectrum(const std::array<T, 4u> &input) {
  using namespace rund::compute;
  const auto make = [](const Backend selected) {
    return on(rund::node::test_contract::target_for(selected))
        .template map<T>("spectrum-parity", 4u,
                         [](auto value) { return quantize<T>(value); })
        .template matrix<2u, 2u>()
        .template svd<SpectrumVectors::Thin>()
        .compile();
  };
  auto cpu_program = make(Backend::Cpu);
  if (!cpu_program) {
    return 1;
  }
  auto cpu = cpu_program->resident(input);
  if (!cpu) {
    std::fprintf(stderr, "spectrum resident backend=%u error=%.*s\n",
                 static_cast<unsigned>(Backend::Cpu),
                 static_cast<int>(cpu.error().size()), cpu.error().data());
    return 2;
  }
  if (!Warm(*cpu, Backend::Cpu, "spectrum")) {
    return 2;
  }
  const Stats cpu_stats = cpu->stats();
  auto cpu_output = cpu->read_all();
  if (!cpu_output) {
    return 3;
  }
  for (const Backend backend :
       rund::node::test_contract::selected_accelerators()) {
    auto target_program = make(backend);
    if (!target_program) {
      return 1;
    }
    auto target = target_program->resident(input);
    if (!target) {
      std::fprintf(stderr, "spectrum resident backend=%u error=%.*s\n",
                   static_cast<unsigned>(backend),
                   static_cast<int>(target.error().size()),
                   target.error().data());
      return 2;
    }
    if (!Warm(*target, backend, "spectrum")) {
      return 2;
    }
    const Stats target_stats = target->stats();
    auto target_output = target->read_all();
    if (!target_output || *cpu_output != *target_output ||
        cpu_stats.graph_hash != target_stats.graph_hash ||
        cpu_stats.output_hash != target_stats.output_hash) {
      return 3;
    }
  }
  return 0;
}

template <class T> [[nodiscard]] int CheckType() {
  using rund::compute::Backend;
  const T zero = T::zero();
  const T half = Half<T>();
  const std::array<T, 4u> identity{half, zero, zero, half};
  for (const Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    auto product =
        rund::compute::on(rund::node::test_contract::target_for(backend),
                          identity)
            .map("fixed-product",
                 rund::compute::capture(
                     [](auto value, auto constant) {
                       return rund::compute::quantize<T>(value * constant);
                     },
                     half))
            .collect();
    if (!product) {
      std::fprintf(stderr, "fixed product failed backend=%u error=%.*s\n",
                   static_cast<unsigned>(backend),
                   static_cast<int>(product.error().size()),
                   product.error().data());
      return 30;
    }
    if (*product != std::vector<T>{Quarter<T>(), zero, zero, Quarter<T>()}) {
      std::fprintf(
          stderr,
          "fixed product mismatch backend=%u first=%lld expected=%lld\n",
          static_cast<unsigned>(backend),
          static_cast<long long>((*product)[0].raw()),
          static_cast<long long>(Quarter<T>().raw()));
      return 30;
    }
  }
  if (const int factor = CheckFactor(identity); factor != 0) {
    return factor;
  }
  if (const int solve = CheckSolve(identity); solve != 0) {
    return 10 + solve;
  }
  if (const int spectrum = CheckSpectrum(identity); spectrum != 0) {
    return 20 + spectrum;
  }
  return 0;
}

} // namespace

int CheckLane32() { return CheckType<rund::compute::Fixed<1, 31>>(); }

int CheckLane64() { return CheckType<rund::compute::Fixed<1, 63>>(); }

} // namespace rund::node::test_contract::numeric
