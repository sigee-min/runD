#pragma once

#include <rund/compute.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

namespace rund_node_graph_services {

using rund::compute::Backend;
using rund::compute::graph::Info;
using rund::compute::graph::Operation;
using rund::compute::graph::Visibility;
using rund::compute::resource::AccessMode;

struct ExecutionEvidence final {
  std::uint64_t graph_hash{};
  std::uint64_t output_hash{};
};

struct CanonicalReference final {
  bool initialized{};
  Info graph{};
  ExecutionEvidence execution{};
};

struct CrossBackendReferences final {
  CanonicalReference asynchronous{};
  std::array<CanonicalReference, 4u> memory{};
  std::array<CanonicalReference, 2u> mixed_fixed{};
  std::array<CanonicalReference, 2u> fixed_policy{};
  std::array<CanonicalReference, 2u> fixed_constant{};
};

[[nodiscard]] bool ValidResourceGraph(const Info &);
[[nodiscard]] bool SameGraph(const Info &, const Info &);
[[nodiscard]] bool MatchReference(CanonicalReference &, const Info &,
                                  ExecutionEvidence);
[[nodiscard]] bool CheckMemoryReuse(rund::compute::Device &, Backend,
                                    std::array<CanonicalReference, 4u> &);
[[nodiscard]] bool CheckAsynchronous(rund::compute::Device &, Backend,
                                     CanonicalReference &);
[[nodiscard]] bool CheckMixedFixed(rund::compute::Device &, Backend,
                                   std::array<CanonicalReference, 2u> &);
[[nodiscard]] bool CheckFixedCache(rund::compute::Device &, Backend,
                                   CrossBackendReferences &);

[[nodiscard]] bool ValidResourcePlan();
[[nodiscard]] bool ValidBoundaryPlan();
[[nodiscard]] bool ValidMemoryBirth();
[[nodiscard]] bool ValidMemoryPlan();
[[nodiscard]] int CheckPolicy(rund::compute::Device *);
[[nodiscard]] int CheckIdentity(rund::compute::Device *,
                                rund::compute::ProgramCache *);
[[nodiscard]] int CheckBounded(rund::compute::Device *,
                               rund::compute::ProgramCache *);
[[nodiscard]] int CheckCache(rund::compute::Device *);
[[nodiscard]] int CheckEmpty(rund::compute::Device *);
[[nodiscard]] int CheckBackends();

} // namespace rund_node_graph_services
