#include "model.hpp"

#include "../../../support.hpp"

#include <rund/compute.hpp>
#include <rund/compute/pipeline.hpp>
#include <rund/session.hpp>

#include <cstdio>
#include <utility>

namespace runtime_compute_pipeline {

template <class T>
bool ReadExact(const rund::compute::Pipeline &pipeline,
               const rund::compute::Buffer<T> &buffer,
               const std::span<T> output) {
  return static_cast<bool>(pipeline.read(buffer, output));
}

template bool
ReadExact<std::int32_t>(const rund::compute::Pipeline &,
                        const rund::compute::Buffer<std::int32_t> &,
                        std::span<std::int32_t>);
template bool
ReadExact<std::uint32_t>(const rund::compute::Pipeline &,
                         const rund::compute::Buffer<std::uint32_t> &,
                         std::span<std::uint32_t>);
template bool ReadExact<rund::compute::Fixed<20u, 44u>>(
    const rund::compute::Pipeline &,
    const rund::compute::Buffer<rund::compute::Fixed<20u, 44u>> &,
    std::span<rund::compute::Fixed<20u, 44u>>);

int Run() {
  auto device = rund::compute::open(rund::compute::Target::cpu(2u));
  if (!device) {
    return 1;
  }
  rund::Session session{};
  if (!session.open(rund::node::test_contract::Options())) {
    return 2;
  }

  struct Contract {
    const char *name;
    int base;
    int (*run)(rund::Session &, rund::compute::Device &);
  };
  constexpr Contract contracts[]{
      {"poll/wait/await", 100, Lifecycle},
      {"claims", 200, Claims},
      {"cancellation", 300, Cancellation},
      {"semantic status", 400, Semantic},
      {"views", 450, View},
      {"state publication", 500, Publish},
      {"state failure", 600, Failure},
      {"state cancel", 700, StateCancel},
      {"nested work totals", 750, NestedWorkTotals},
  };
  for (const Contract &contract : contracts) {
    if (const int result = contract.run(session, *device); result != 0) {
      std::fprintf(stderr, "runtime pipeline %s=%d\n", contract.name, result);
      return contract.base + result;
    }
  }
  if (!session.close()) {
    return 3;
  }
  if (const int result = Close(*device); result != 0) {
    std::fprintf(stderr, "runtime pipeline close=%d\n", result);
    return 800 + result;
  }
  return 0;
}

} // namespace runtime_compute_pipeline
