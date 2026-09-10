#include "../model.hpp"

#include <cstdio>

namespace rund_node_collective_modes::bounded {
namespace {

template <class T>
[[nodiscard]] bool CheckAggregateFor(const rund::compute::Backend backend,
                                     DomainEvidence &evidence) {
  using namespace rund::compute;
  auto input = TailValues<T>();
  auto target = flow_on(backend, Target::cpu(2u));
  auto program =
      std::move(target)
          .template map<T>("mode-bounded-input", input.size(),
                           [](auto value) { return Store<T>(value); })
          .filter([](auto value) { return value != Zero<T>(); })
          .branch([](auto values) {
            const auto maximum = values.reduce(Reduce::Max);
            return outputs(
                values.map("mode-bounded-map",
                           [](auto value) { return Store<T>(value); }),
                values.scan(Scan::InclusiveSum),
                values.scan(Scan::ExclusiveSum), values.reduce(Reduce::Sum),
                values.reduce(Reduce::Min), maximum, values.sort(),
                values.argsort(), values.count(),
                maximum.map("mode-scalar-map",
                            [](auto value) { return Store<T>(value); }));
          })
          .compile();
  if (!program) {
    std::fprintf(
        stderr, "compute modes compile backend=%u family=bounded reason=%.*s\n",
        static_cast<unsigned>(backend),
        static_cast<int>(program.error().size()), program.error().data());
    return false;
  }
  auto job = program->resident(input);
  return job && SameSuccess(*job, backend, "bounded", evidence.bounded);
}

} // namespace

bool CheckAggregate(const rund::compute::Backend backend,
                    DomainEvidence &evidence, const Domain domain) {
  switch (domain) {
  case Domain::I32:
    return CheckAggregateFor<std::int32_t>(backend, evidence);
  case Domain::U32:
    return CheckAggregateFor<std::uint32_t>(backend, evidence);
  case Domain::I64:
    return CheckAggregateFor<std::int64_t>(backend, evidence);
  case Domain::U64:
    return CheckAggregateFor<std::uint64_t>(backend, evidence);
  case Domain::Fixed16x16:
    return CheckAggregateFor<rund::compute::Fixed<16, 16>>(backend, evidence);
  case Domain::Fixed20x44:
    return CheckAggregateFor<rund::compute::Fixed<20, 44>>(backend, evidence);
  case Domain::Lane32:
  case Domain::Lane64:
    return false;
  }
  return false;
}

} // namespace rund_node_collective_modes::bounded
