#include "../model.hpp"

#include "../../../target/selection.hpp"

#include <array>
#include <cstdio>

namespace rund_node_memory_contract {

int CheckSortRunMemory(const rund::compute::Backend backend) {
  using namespace rund::compute;
  const std::array<std::uint32_t, 4u> input{3u, 1u, 4u, 2u};
  auto program = on(rund::node::test_contract::target_for(backend))
                     .map<std::uint32_t>("memory-sort", input.size(),
                                         [](auto value) { return value; })
                     .sort()
                     .compile();
  if (!program) {
    std::fprintf(stderr, "sort memory backend=%u compile=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(program.error().size()),
                 program.error().data());
    return 1;
  }
  auto job = program->resident(input);
  if (!job) {
    std::fprintf(stderr, "sort memory backend=%u prepare=%.*s\n",
                 static_cast<unsigned>(backend),
                 static_cast<int>(job.error().size()), job.error().data());
    return 2;
  }
  const bool retained = backend == Backend::Metal || backend == Backend::Vulkan;
  const MemoryCounter prepared = job->memory().staging;
  if (!job->run()) {
    return 3;
  }
  const MemoryCounter first = job->memory().staging;
  if (first.current != prepared.current ||
      (retained ? (prepared.current == 0u || first.peak != prepared.peak ||
                   first.cumulative != prepared.cumulative)
                : (first.peak < prepared.peak || first.peak <= first.current ||
                   first.cumulative <= prepared.cumulative))) {
    std::fprintf(stderr,
                 "sort memory backend=%u prepared=%llu/%llu/%llu "
                 "first=%llu/%llu/%llu\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned long long>(prepared.current),
                 static_cast<unsigned long long>(prepared.peak),
                 static_cast<unsigned long long>(prepared.cumulative),
                 static_cast<unsigned long long>(first.current),
                 static_cast<unsigned long long>(first.peak),
                 static_cast<unsigned long long>(first.cumulative));
    return 4;
  }
  if (!job->run()) {
    return 5;
  }
  const MemoryCounter second = job->memory().staging;
  if (second.current != first.current ||
      (retained ? (second.cumulative != first.cumulative ||
                   second.reused != first.reused)
                : (second.peak != first.peak ||
                   second.cumulative <= first.cumulative ||
                   second.reused <= first.reused))) {
    return 6;
  }
  return 0;
}

} // namespace rund_node_memory_contract
