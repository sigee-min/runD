#include "../local.hpp"

#include "src/compute/pipeline/plan/contract.hpp"
#include "src/compute/pipeline/plan/publication.hpp"

namespace rund::node::test_contract::window {

[[nodiscard]] bool PublicationFingerprintV3Golden() {
  using namespace rund::compute::detail;
  const auto view =
      [](const std::uint32_t ordinal, const std::uint64_t backing_bytes,
         const std::uint64_t offset_bytes, const std::uint64_t count,
         const std::uint64_t stride_bytes, const std::uint32_t usage) {
        return PipelinePublicationViewPlan{
            .identity =
                PipelinePublicationViewIdentity{
                    .backing_bytes = backing_bytes,
                    .offset_bytes = offset_bytes,
                    .count = count,
                    .stride_bytes = stride_bytes,
                    .element_bytes = sizeof(std::uint32_t),
                    .resource_ordinal = ordinal,
                    .usage = usage,
                },
            .type = Type::U32,
        };
      };

  PipelineTerminalPublicationPlan terminal{
      .sources =
          {
              view(2u, 64u, 0u, 1u, 4u, rund::kernel::kResidentUsageRead),
              view(3u, 64u, 0u, 1u, 4u, rund::kernel::kResidentUsageRead),
              view(4u, 64u, 0u, 1u, 4u, rund::kernel::kResidentUsageRead),
          },
      .target =
          PipelinePublicationTargetPlan{
              .view =
                  view(7u, 64u, 8u, 1u, 8u, rund::kernel::kResidentUsageWrite),
          },
      .state = 1u,
      .output = {.value = 2u},
  };
  PipelineWindowControl terminal_control{.final = 2u};
  PipelineHash terminal_hash{};
  terminal_hash.number(1u);
  if (!mix_pipeline_publication_public_identity(terminal_hash, terminal,
                                                terminal_control)) {
    return false;
  }
  constexpr Fingerprint expected_terminal{
      .hi = 0xb5087f04bc866a06ull,
      .lo = 0x140961c5d5e8c583ull,
  };
  if (terminal_hash.finish() != expected_terminal) {
    return false;
  }

  // Public v3 serializes the selected canonical source in the source-ordinal
  // field. It deliberately does not add the private three-bank/final shape.
  PipelineTerminalPublicationPlan compatible = terminal;
  compatible.sources[0] =
      view(99u, 128u, 16u, 2u, 12u, rund::kernel::kResidentUsageRead);
  compatible.sources[1] = terminal.sources[2];
  PipelineWindowControl compatible_control{.final = 1u};
  PipelineHash compatible_hash{};
  compatible_hash.number(1u);
  if (!mix_pipeline_publication_public_identity(compatible_hash, compatible,
                                                compatible_control) ||
      compatible_hash.finish() != expected_terminal) {
    return false;
  }

  PipelineWindowPublicationPlan window{
      .source = view(9u, 16u, 0u, 4u, 4u, rund::kernel::kResidentUsageRead),
      .target =
          PipelinePublicationTargetPlan{
              .view = view(11u, 80u, 8u, 16u, 4u,
                           rund::kernel::kResidentUsageWrite),
          },
      .state = 2u,
      .output = {.value = 3u},
  };
  PipelineWindowControl window_control{
      .count = view(10u, 12u, 4u, 1u, 4u, rund::kernel::kResidentUsageRead),
      .maximum = 16u,
      .tile = 4u,
  };
  PipelineHash mixed_hash{};
  mixed_hash.number(2u);
  if (!mix_pipeline_publication_public_identity(mixed_hash, terminal,
                                                terminal_control) ||
      !mix_pipeline_publication_public_identity(mixed_hash, window,
                                                window_control)) {
    return false;
  }
  constexpr Fingerprint expected_mixed{
      .hi = 0xc9ed4d382b576784ull,
      .lo = 0x8c532fa716b8c443ull,
  };
  return mixed_hash.finish() == expected_mixed;
}

} // namespace rund::node::test_contract::window
