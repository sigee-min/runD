#pragma once

#include "../backing.hpp"
#include "../route.hpp"
#include "src/compute/virtual/graph_resident/workload.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rund_node_test_virtual::product::graph_resident {

using Workload = rund::compute::graph_resident_workload::Spec;
using Variant = Workload::Variant;
inline constexpr std::size_t InputCount = Workload::InputCount;
inline constexpr std::size_t StageCount = Workload::StageCount;
inline constexpr std::size_t FrameElements = Workload::FrameElements;
inline constexpr std::size_t PageCount = Workload::PageCount;
inline constexpr std::size_t TailElements = Workload::TailElements;
inline constexpr std::size_t ElementCount = Workload::ElementCount;
inline constexpr std::size_t LeafCount = Workload::LeafCount;
inline constexpr std::size_t BatchCount = Workload::BatchCount;
inline constexpr std::size_t RunCount = Workload::RunCount;
inline constexpr std::size_t InternalOwnerCount = Workload::InternalOwnerCount;
inline constexpr std::size_t TotalClassCount = Workload::TotalClassCount;

using Program = Workload::Program;
using Pipeline = rund::compute::VirtualPipeline<std::uint64_t(
    std::uint64_t, std::uint64_t, std::uint64_t)>;

struct U32Workload final {
  using Scalar = std::uint32_t;
  using Program = rund::compute::Program<Scalar(Scalar, Scalar, Scalar)>;

  template <std::uint32_t First, std::size_t Count, class Expression>
  [[nodiscard]] static constexpr auto add_stage(Expression value) {
    if constexpr (Count == 1u) {
      return value + First;
    } else {
      constexpr std::size_t left = Count / 2u;
      return add_stage<First, left>(value) +
             add_stage<static_cast<std::uint32_t>(First + left), Count - left>(
                 value);
    }
  }

  [[nodiscard]] static constexpr Scalar
  input_value(const std::size_t input, const std::size_t index) noexcept {
    return input < InputCount
               ? static_cast<Scalar>(index * Workload::Steps[input] +
                                     Workload::Seeds[input])
               : 0u;
  }

  [[nodiscard]] static constexpr Scalar
  stage_value(const Scalar value, const Scalar first) noexcept {
    constexpr std::uint64_t literal_sum = LeafCount * (LeafCount + 1u) / 2u;
    return static_cast<Scalar>(
        static_cast<std::uint64_t>(value) * LeafCount + literal_sum +
        static_cast<std::uint64_t>(first - 1u) * LeafCount);
  }

  [[nodiscard]] static constexpr Scalar
  expected_value(const std::size_t index) noexcept {
    const Scalar x = stage_value(input_value(0u, index), 1u);
    const Scalar y = stage_value(input_value(1u, index),
                                 static_cast<Scalar>(LeafCount + 1u));
    const Scalar z = static_cast<Scalar>(
        static_cast<std::uint64_t>(
            stage_value(x, static_cast<Scalar>(2u * LeafCount + 1u))) +
        stage_value(y, static_cast<Scalar>(3u * LeafCount + 1u)));
    const Scalar w = stage_value(input_value(2u, index),
                                 static_cast<Scalar>(4u * LeafCount + 1u));
    return static_cast<Scalar>(
        static_cast<std::uint64_t>(
            stage_value(z, static_cast<Scalar>(5u * LeafCount + 1u))) +
        stage_value(w, static_cast<Scalar>(6u * LeafCount + 1u)));
  }

  [[nodiscard]] static std::uint64_t expected_hash() noexcept {
    std::array<Scalar, ElementCount> values{};
    for (std::size_t index = 0u; index < ElementCount; ++index) {
      values[index] = expected_value(index);
    }
    return ::rund::node::hash_detail::HashBytes(values.data(),
                                                values.size() * sizeof(Scalar));
  }
};

using U32Program = U32Workload::Program;
using U32Pipeline = rund::compute::VirtualPipeline<std::uint32_t(
    std::uint32_t, std::uint32_t, std::uint32_t)>;

struct Observation final {
  rund::compute::Status status =
      rund::compute::Status::fail(rund::compute::Reason::CompletionInvalid);
  std::uint32_t status_reason{};
  const char *device_vsm_prepare_reason{};
  rund::compute::Stats before{};
  rund::compute::Stats after{};
  std::uint64_t version_before{};
  std::uint64_t version_after{};
  std::uint64_t output_hash{};
  std::uint64_t proof_digest{};
  std::uint8_t proof_scalar{};
  std::uint8_t proof_domain{};
  std::uint32_t proof_element_bytes{};
  std::uint64_t native_proof_hi{};
  std::uint64_t native_proof_lo{};
  std::uint64_t native_generation{};
  std::uint64_t native_nonce{};
  std::uint64_t native_completed_ns{};
  std::uint64_t native_kernel_ns{};
  std::uint64_t native_kernel_samples{};
  std::uint64_t native_submit_wait_ns{};
  bool native_may_write{};
  bool output_match{};
  bool graph_resident{};
  RouteKind route_kind{RouteKind::Unknown};
  std::uint32_t accepted_owner_mask{};
  std::uint32_t accepted_owner_count{};
  std::uint64_t native_submit_count{};
  std::uint64_t epoch_submit_count{};
  std::uint64_t dispatch_count{};
  std::uint64_t tile_dispatch_count{};
  std::uint64_t final_count{};
  std::uint64_t generated_pages{};
  std::uint64_t completed_pages{};
  std::uint64_t forecasted_pages{};
  std::uint64_t promoted_pages{};
  std::uint64_t drained_pages{};
  std::uint64_t persisted_pages{};
  std::uint64_t wavefront_steps{};
  std::uint64_t gpu_read_bytes{};
  std::uint64_t gpu_write_bytes{};
  bool host_service{};
  bool quarantined{};
  bool staged_output{};
  std::uint32_t resource_count{};
  std::uint32_t internal_owners{};
  bool alias_reuse{};
};

struct Case final {
  Variant variant{Variant::Ordinary};
  std::vector<std::uint64_t> first_values;
  std::vector<std::uint64_t> second_values;
  std::vector<std::uint64_t> third_values;
  std::vector<std::uint64_t> expected;
  std::array<std::shared_ptr<rund::compute::VirtualBacking>, InputCount>
      inputs{};
  std::shared_ptr<rund::compute::VirtualBacking> output;
  Pipeline pipeline;
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state;
};

struct U32Case final {
  std::vector<std::uint32_t> first_values;
  std::vector<std::uint32_t> second_values;
  std::vector<std::uint32_t> third_values;
  std::vector<std::uint32_t> expected;
  std::array<std::shared_ptr<rund::compute::VirtualBacking>, InputCount>
      inputs{};
  std::shared_ptr<rund::compute::VirtualBacking> output;
  U32Pipeline pipeline;
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state;
};

struct Preparation final {
  std::unique_ptr<Case> value{};
  int reason{};
};

struct U32Preparation final {
  std::unique_ptr<U32Case> value{};
  int reason{};
};

[[nodiscard]] bool
seed_backing(const std::shared_ptr<rund::compute::VirtualBacking> &,
             std::span<const std::byte>) noexcept;
[[nodiscard]] rund::compute::Result<Program>
build_program(const rund::compute::Device &, Variant = Variant::Ordinary);
[[nodiscard]] bool validate_program(const Program &,
                                    Variant = Variant::Ordinary) noexcept;
[[nodiscard]] Preparation prepare_case(const rund::compute::Device &,
                                       Variant = Variant::Ordinary);
[[nodiscard]] bool run_case(Case &, std::array<Observation, RunCount> &,
                            rund::compute::Device &, rund::compute::Backend);
[[nodiscard]] bool capture_run(Case &, Observation &) noexcept;
[[nodiscard]] bool validate_case(const Case &, rund::compute::Backend,
                                 std::span<const Observation>) noexcept;
[[nodiscard]] bool validate_dynamic_case(const Case &, rund::compute::Backend,
                                         const Observation &) noexcept;
[[nodiscard]] rund::compute::Result<U32Program>
build_u32_program(const rund::compute::Device &);
[[nodiscard]] bool validate_u32_program(const U32Program &) noexcept;
[[nodiscard]] U32Preparation prepare_u32_case(const rund::compute::Device &);
[[nodiscard]] bool run_u32_case(U32Case &, std::array<Observation, RunCount> &,
                                rund::compute::Device &,
                                rund::compute::Backend);
[[nodiscard]] bool capture_u32_run(U32Case &, Observation &) noexcept;
[[nodiscard]] bool validate_u32_case(const U32Case &, rund::compute::Backend,
                                     std::span<const Observation>) noexcept;

} // namespace rund_node_test_virtual::product::graph_resident
