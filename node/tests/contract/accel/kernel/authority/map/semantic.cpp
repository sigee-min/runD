#include "../map.hpp"

#include "src/accel/graph/step/map_semantic.hpp"

#include <kernel/program/compute/dsl.hpp>
#include <kernel/program/compute/lowering/emission.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace node_accel_contract {

bool MapArithmeticMeaningIsCanonical() {
  using namespace rund::node::accel::detail;
  using rund::kernel::ComputeApi;
  using rund::kernel::compute_lowering_detail::LowerRetainedComputeArtifact;
  constexpr std::array immediates{
      0u, 1u, 3u, 4u, std::numeric_limits<std::uint32_t>::max()};
  for (const auto api : {ComputeApi::Cpu, ComputeApi::Metal, ComputeApi::Vulkan}) {
    for (const std::uint32_t immediate : immediates) {
      std::uint32_t input[1]{};
      std::uint32_t output[1]{};
      const auto body = rund::compute_dsl::bind(1u)
                            .u32()
                            .read<"input">(input)
                            .write<"output">(output);
      for (const bool multiply : {false, true}) {
        const auto op = rund::compute_dsl::def("map-arithmetic-meaning")
                            .on(body)
                            .map([=](auto i, auto bindings) {
                              const auto value =
                                  bindings.template read<"input">()[i];
                              // Compute Map emission materializes its logical
                              // index after the reads, including scalar Maps.
                              static_cast<void>(rund::compute_dsl::detail::TypedIndex(
                                  value, rund::compute_dsl::detail::ScalarMode::U32));
                              bindings.template write<"output">()[i] =
                                  multiply ? value * immediate
                                           : value + immediate;
                            });
        auto retained = LowerRetainedComputeArtifact(op.ir(), api);
        if (!retained.artifact.ok || !retained.input.ok) {
          return false;
        }
        const auto meaning =
            step::BuildMapSemantic(retained.artifact, retained.input);
        const auto expected = multiply ? MapSemanticKind::MulWrapU32Immediate
                                       : MapSemanticKind::AddWrapU32Immediate;
        if (meaning.kind != expected || meaning.immediate != immediate ||
            !meaning.recurrence_total) {
          std::fprintf(stderr,
                       "map meaning api=%u multiply=%u immediate=%u "
                       "kind=%u expected=%u\n",
                       static_cast<unsigned>(api), multiply, immediate,
                       static_cast<unsigned>(meaning.kind),
                       static_cast<unsigned>(expected));
          return false;
        }
        auto rejected = retained.input;
        rejected.ok = false;
        auto invalid = step::BuildMapSemantic(retained.artifact, rejected);
        if (invalid.kind != MapSemanticKind::Unknown || invalid.recurrence_total) {
          return false;
        }
        rejected = retained.input;
        rejected.key.op_hash_lo ^= 1u;
        invalid = step::BuildMapSemantic(retained.artifact, rejected);
        if (invalid.kind != MapSemanticKind::Unknown || invalid.recurrence_total) {
          return false;
        }
      }
    }
  }
  return true;
}

} // namespace node_accel_contract
