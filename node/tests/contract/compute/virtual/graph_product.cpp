#include "product/graph_resident/internal.hpp"
#include "product/graph_resident_host/internal.hpp"
#include "product/local.hpp"

#include "../../target/selection.hpp"

#include <array>
#include <cstdio>

namespace rund_node_test_virtual::product {

int CheckProductGraphResidentHost() {
  using namespace rund::compute;
  if (!rund::node::test_contract::backend_selected(Backend::Cpu)) {
    return 0;
  }
  auto opened = open(rund::node::test_contract::target_for(Backend::Cpu));
  if (!opened) {
    std::fprintf(stderr, "compute virtual GraphResident Host open reason=%u\n",
                 static_cast<unsigned>(opened.reason()));
    return 2;
  }
  auto prepared = graph_resident_host::prepare_case(*opened);
  if (!prepared.value) {
    return 10 + prepared.reason;
  }
  auto &test_case = *prepared.value;
  graph_resident_host::Observation observation{};
  if (!graph_resident_host::run_case(test_case, observation, *opened)) {
    return 20;
  }
  return graph_resident_host::validate_case(test_case, observation) ? 0 : 21;
}

int CheckProductGraphResident(const rund::compute::Backend backend) {
  using namespace rund::compute;
  if (backend == Backend::Cpu) {
    return 0;
  }
  if (backend != Backend::Metal && backend != Backend::Vulkan) {
    return 1;
  }
  auto opened =
      rund::compute::open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    std::fprintf(
        stderr, "compute virtual GraphResident open backend=%u reason=%u\n",
        static_cast<unsigned>(backend), static_cast<unsigned>(opened.reason()));
    return opened.reason() == Reason::AdapterUnavailable ? 2 : 1;
  }
  const auto run_variant = [&](const graph_resident::Variant variant) {
    auto prepared = graph_resident::prepare_case(*opened, variant);
    if (!prepared.value) {
      return 10 + prepared.reason;
    }
    auto &test_case = *prepared.value;
    std::array<graph_resident::Observation, graph_resident::RunCount>
        observations{};
    if (!graph_resident::run_case(test_case, observations, *opened, backend)) {
      return 20;
    }
    return graph_resident::validate_case(test_case, backend, observations) ? 0
                                                                           : 21;
  };
  const int ordinary = run_variant(graph_resident::Variant::Ordinary);
  return ordinary == 0 ? run_variant(graph_resident::Variant::AddSatUnsigned)
                       : ordinary;
}

} // namespace rund_node_test_virtual::product

int RunComputeVirtualGraphResidencyProductContract() {
  const int host_result =
      rund_node_test_virtual::product::CheckProductGraphResidentHost();
  if (host_result != 0) {
    std::fprintf(stderr, "compute virtual GraphResident Host result=%d\n",
                 host_result);
    return 100 + host_result;
  }
  if (rund::node::test_contract::backend_selected(rund::compute::Backend::Cpu)) {
    const int scratch_result =
        rund_node_test_virtual::product::CheckProductGraphForecastWindow(
            rund::compute::Backend::Cpu);
    if (scratch_result != 0) {
      std::fprintf(stderr, "compute virtual CPU Graph scratch result=%d\n",
                   scratch_result);
      return 200 + scratch_result;
    }
  }
  for (const rund::compute::Backend backend :
       rund::node::test_contract::selected_accelerators()) {
    const int multi_host_result =
        rund_node_test_virtual::product::CheckProductGraphMultiHostWavefront(
            backend);
    if (multi_host_result != 0) {
      std::fprintf(stderr,
                   "compute virtual multi Host graph product backend=%u "
                   "result=%d\n",
                   static_cast<unsigned>(backend), multi_host_result);
      return static_cast<int>(backend) * 1000 + multi_host_result;
    }
    const int result =
        rund_node_test_virtual::product::CheckProductGraphWavefront(backend);
    if (result != 0) {
      std::fprintf(stderr,
                   "compute virtual graph product backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return static_cast<int>(backend) * 1000 + result;
    }
    const int host_result =
        rund_node_test_virtual::product::CheckProductGraphHostWavefront(
            backend);
    if (host_result != 0) {
      std::fprintf(stderr,
                   "compute virtual Host graph product backend=%u result=%d\n",
                   static_cast<unsigned>(backend), host_result);
      return static_cast<int>(backend) * 1000 + host_result;
    }
    const int forecast_result =
        rund_node_test_virtual::product::CheckProductGraphForecastWindow(
            backend);
    if (forecast_result != 0) {
      std::fprintf(stderr,
                   "compute virtual Forecast window backend=%u result=%d\n",
                   static_cast<unsigned>(backend), forecast_result);
      return static_cast<int>(backend) * 1000 + forecast_result;
    }
    const int persist_result =
        rund_node_test_virtual::product::CheckProductGraphPersistRing(backend);
    if (persist_result != 0) {
      std::fprintf(stderr,
                   "compute virtual Graph Persist ring backend=%u result=%d\n",
                   static_cast<unsigned>(backend), persist_result);
      return static_cast<int>(backend) * 1000 + persist_result;
    }
    const int pointwise_result =
        rund_node_test_virtual::product::CheckProductGraphPointwise(backend);
    if (pointwise_result != 0) {
      std::fprintf(stderr,
                   "compute virtual pointwise graph product backend=%u "
                   "result=%d\n",
                   static_cast<unsigned>(backend), pointwise_result);
      return static_cast<int>(backend) * 1000 + pointwise_result;
    }
    const int wide_pointwise_result =
        rund_node_test_virtual::product::CheckProductGraphPointwiseWideHost(
            backend);
    if (wide_pointwise_result != 0) {
      std::fprintf(stderr,
                   "compute virtual wide-input pointwise graph product "
                   "backend=%u result=%d\n",
                   static_cast<unsigned>(backend), wide_pointwise_result);
      return static_cast<int>(backend) * 1000 + wide_pointwise_result;
    }
    const int wide_device_result =
        rund_node_test_virtual::product::CheckProductGraphPointwiseWideDevice(
            backend);
    if (wide_device_result != 0) {
      std::fprintf(stderr,
                   "compute virtual wide-input GPU Graph pointwise product "
                   "backend=%u result=%d\n",
                   static_cast<unsigned>(backend), wide_device_result);
      return static_cast<int>(backend) * 1000 + wide_device_result;
    }
    const int deep_device_result =
        rund_node_test_virtual::product::CheckProductGraphPointwiseDeepDevice(
            backend);
    if (deep_device_result != 0) {
      std::fprintf(stderr,
                   "compute virtual deep GPU Graph pointwise product "
                   "backend=%u result=%d\n",
                   static_cast<unsigned>(backend), deep_device_result);
      return static_cast<int>(backend) * 1000 + deep_device_result;
    }
    const int deeper_device_result =
        rund_node_test_virtual::product::CheckProductGraphPointwiseDeeperDevice(
            backend);
    if (deeper_device_result != 0) {
      std::fprintf(stderr,
                   "compute virtual deeper GPU Graph pointwise product "
                   "backend=%u result=%d\n",
                   static_cast<unsigned>(backend), deeper_device_result);
      return static_cast<int>(backend) * 1000 + deeper_device_result;
    }
    const int frontier_device_result = rund_node_test_virtual::product::
        CheckProductGraphPointwiseFrontierDevice(backend);
    if (frontier_device_result != 0) {
      std::fprintf(stderr,
                   "compute virtual frontier GPU Graph pointwise product "
                   "backend=%u result=%d\n",
                   static_cast<unsigned>(backend), frontier_device_result);
      return static_cast<int>(backend) * 1000 + frontier_device_result;
    }
    const int resident_result =
        rund_node_test_virtual::product::CheckProductGraphResident(backend);
    if (resident_result != 0) {
      std::fprintf(stderr,
                   "compute virtual GraphResident product backend=%u "
                   "result=%d\n",
                   static_cast<unsigned>(backend), resident_result);
      return static_cast<int>(backend) * 1000 + resident_result;
    }
    const int depth_six_result = rund_node_test_virtual::product::
        CheckProductGraphPointwiseDepthSixDevice(backend);
    if (depth_six_result != 0) {
      std::fprintf(stderr,
                   "compute virtual six-stage GPU Graph pointwise product "
                   "backend=%u result=%d\n",
                   static_cast<unsigned>(backend), depth_six_result);
      return static_cast<int>(backend) * 1000 + depth_six_result;
    }
    const int depth_seven_result = rund_node_test_virtual::product::
        CheckProductGraphPointwiseDepthSevenDevice(backend);
    if (depth_seven_result != 0) {
      std::fprintf(stderr,
                   "compute virtual seven-stage GPU Graph pointwise product "
                   "backend=%u result=%d\n",
                   static_cast<unsigned>(backend), depth_seven_result);
      return static_cast<int>(backend) * 1000 + depth_seven_result;
    }
    const int multi_pointwise_result =
        rund_node_test_virtual::product::CheckProductGraphPointwiseMulti(
            backend);
    if (multi_pointwise_result != 0) {
      std::fprintf(stderr,
                   "compute virtual multi-input pointwise graph product "
                   "backend=%u result=%d\n",
                   static_cast<unsigned>(backend), multi_pointwise_result);
      return static_cast<int>(backend) * 1000 + multi_pointwise_result;
    }
  }
  return 0;
}
