#include "local.hpp"

#include "src/compute/device/residency/registry.hpp"

#include <array>
#include <cstddef>
#include <limits>

namespace rund_node_test_pipeline_residency {

int CheckAuthorityTransform() {
  using namespace rund::compute::detail::residency;
  const auto never = std::numeric_limits<std::uint64_t>::max();
  // Input and output regions participate in one transform lease but retain
  // independent identities. Wrong role/tier and overflowing dirty extents are
  // rejected before either region mutates. Transient partials can migrate or
  // retire, but can never enter backing writeback or terminal drain.
  Authority transform;
  std::uint32_t transform_input = 99u;
  std::uint32_t transform_intermediate = 99u;
  std::uint32_t transform_output = 99u;
  if (!transform.register_frames(FrameTier::Device, FrameRole::Input, 2u,
                                 transform_input) ||
      !transform.register_frames(FrameTier::Device, FrameRole::Intermediate,
                                 2u, transform_intermediate) ||
      !transform.register_frames(FrameTier::Device, FrameRole::Output, 2u,
                                 transform_output)) {
    return 45;
  }
  const std::array transform_inputs{
      CacheUse{.key = {.backing = 60u,
                       .version = 1u,
                       .materialization_hi = 9u,
                       .page = 0u},
               .access = Access::Read,
               .next_use = 2u},
      CacheUse{.key = {.backing = 60u,
                       .version = 1u,
                       .materialization_hi = 9u,
                       .page = 1u},
               .access = Access::Read,
               .next_use = 3u},
  };
  const std::array transform_outputs{
      CacheUse{.key = {.backing = 61u,
                       .version = 1u,
                       .materialization_hi = 9u,
                       .page = 0u,
                       .domain = CacheDomain::Transient},
               .access = Access::Write,
               .next_use = never,
               .dirty = {.offset = 0u, .bytes = 8u}},
      CacheUse{.key = {.backing = 61u,
                       .version = 1u,
                       .materialization_hi = 9u,
                       .page = 1u,
                       .domain = CacheDomain::Transient},
               .access = Access::Write,
               .next_use = never,
               .dirty = {.offset = 8u, .bytes = 8u}},
  };
  const FrameRegion input_region{.tier = FrameTier::Device,
                                 .role = FrameRole::Input,
                                 .first = transform_input,
                                 .count = 2u};
  const FrameRegion output_region{.tier = FrameTier::Device,
                                  .role = FrameRole::Output,
                                  .first = transform_output,
                                  .count = 2u};
  const FrameRegion intermediate_region{.tier = FrameTier::Device,
                                        .role = FrameRole::Intermediate,
                                        .first = transform_intermediate,
                                        .count = 2u};
  if (transform
          .begin_transform(transform_inputs, output_region, transform_outputs,
                           input_region)
          .failure != AuthorityFailure::Invalid) {
    return 46;
  }
  std::array<std::uint8_t, 2u> absent{};
  const std::array transform_input_keys{transform_inputs[0].key,
                                        transform_inputs[1].key};
  if (!transform.probe(transform_input_keys, absent, transform_input, 2u) ||
      absent[0] != 0u || absent[1] != 0u) {
    return 47;
  }
  auto overflow_outputs = transform_outputs;
  overflow_outputs[1].dirty = {.offset = never, .bytes = 2u};
  if (transform
          .begin_transform(transform_inputs, input_region, overflow_outputs,
                           output_region)
          .failure != AuthorityFailure::Invalid) {
    return 48;
  }
  const AuthorityResult transformed = transform.begin_transform(
      transform_inputs, input_region, transform_outputs, output_region);
  if (!transformed || transformed.lease.bindings.size() != 4u ||
      !transform.activate(transformed.lease.token) ||
      !transform.complete(transformed.lease.token, true)) {
    return 49;
  }
  const std::array transform_output_keys{transform_outputs[0].key,
                                         transform_outputs[1].key};
  if (transform.begin_writeback(transform_output_keys, transform_output, 2u) ||
      transform.drain_dirty(transform_output, 2u)) {
    return 50;
  }
  const AuthorityResult retire_direct =
      transform.begin_discard(transform_output_keys, transform_output, 2u);
  if (!retire_direct || !transform.discard(retire_direct.lease.token)) {
    return 51;
  }

  // A graph stage may change materialization identity while retaining logical
  // page order. Dirty Transient victims retire through Discard, never through
  // a backing Writeback transition, and rollback restores the exact stage-0
  // materialization for the next consumer.
  const std::array intermediate_outputs{
      CacheUse{.key = {.backing = 62u,
                       .version = 1u,
                       .materialization_hi = 10u,
                       .page = 0u,
                       .domain = CacheDomain::Transient},
               .access = Access::Write,
               .next_use = 1u,
               .dirty = {.offset = 0u, .bytes = 4096u}},
      CacheUse{.key = {.backing = 62u,
                       .version = 1u,
                       .materialization_hi = 10u,
                       .page = 1u,
                       .domain = CacheDomain::Transient},
               .access = Access::Write,
               .next_use = 1u,
               .dirty = {.offset = 4096u, .bytes = 4096u}},
  };
  const GraphMaterialization graph_input{
      .resource = 1u,
      .key = {.backing = 60u, .version = 1u, .materialization_hi = 9u},
      .page_bytes = 4096u,
      .page_count = 2u,
  };
  const GraphMaterialization graph_intermediate{
      .resource = 2u,
      .key = {.backing = 62u,
              .version = 1u,
              .materialization_hi = 10u,
              .domain = CacheDomain::Transient},
      .page_bytes = 4096u,
      .page_count = 2u,
  };
  std::array<PageUse, 2u> graph_inputs{};
  std::array<PageUse, 2u> graph_intermediates{};
  for (std::size_t index = 0u; index < graph_inputs.size(); ++index) {
    graph_inputs[index] = PageUse{
        .key = {.resource = 1u, .page = index},
        .access = Access::Read,
        .next_use = 4u + index,
        .pin = {.first_epoch = 0u, .last_epoch = 0u},
        .ready_epoch = 0u,
    };
    graph_intermediates[index] = PageUse{
        .key = {.resource = 2u, .page = index},
        .access = Access::Write,
        .dirty = {.bytes = 4096u},
        .next_use = 1u,
        .pin = {.first_epoch = 0u, .last_epoch = 0u},
        .ready_epoch = 0u,
    };
  }
  auto wrong_pin = graph_inputs;
  wrong_pin[0].pin = {.first_epoch = 1u, .last_epoch = 1u};
  if (transform.begin_graph_transform(wrong_pin, graph_input, input_region,
                                      graph_intermediates, graph_intermediate,
                                      intermediate_region, 0u)) {
    return 52;
  }
  const AuthorityResult prefix = transform.begin_graph_transform(
      graph_inputs, graph_input, input_region, graph_intermediates,
      graph_intermediate, intermediate_region, 0u);
  if (!prefix || !transform.activate(prefix.lease.token) ||
      !transform.complete(prefix.lease.token, true)) {
    return 52;
  }
  auto replacement_intermediates = intermediate_outputs;
  replacement_intermediates[0].key.materialization_hi = 11u;
  replacement_intermediates[1].key.materialization_hi = 11u;
  const AuthorityResult transient_replacement =
      transform.begin(replacement_intermediates, FrameTier::Device,
                      FrameRole::Intermediate, transform_intermediate, 2u);
  if (!transient_replacement ||
      transient_replacement.lease.transitions.size() != 6u ||
      transient_replacement.lease.transitions[0].kind !=
          TransitionKind::Discard ||
      transient_replacement.lease.transitions[1].kind !=
          TransitionKind::Unmap ||
      !transform.complete(transient_replacement.lease.token, false)) {
    return 53;
  }
  std::array<PageUse, 2u> collective_inputs{};
  std::array<PageUse, 2u> collective_outputs{};
  for (std::size_t index = 0u; index < collective_inputs.size(); ++index) {
    collective_inputs[index] = PageUse{
        .key = {.resource = 2u, .page = index},
        .access = Access::Read,
        .next_use = never,
        .pin = {.first_epoch = 1u, .last_epoch = 1u},
        .prefetch_epoch = 1u,
        .ready_epoch = 1u,
    };
    collective_outputs[index] = PageUse{
        .key = {.resource = 3u, .page = index},
        .access = Access::Write,
        .dirty = {.bytes = 8u},
        .next_use = never,
        .pin = {.first_epoch = 1u, .last_epoch = 1u},
        .prefetch_epoch = 1u,
        .ready_epoch = 1u,
    };
  }
  const GraphMaterialization graph_partial{
      .resource = 3u,
      .key = {.backing = 61u,
              .version = 1u,
              .materialization_hi = 9u,
              .domain = CacheDomain::Transient},
      .page_bytes = 8u,
      .page_count = 2u,
  };
  const AuthorityResult collective = transform.begin_graph_transform(
      collective_inputs, graph_intermediate, intermediate_region,
      collective_outputs, graph_partial, output_region, 1u);
  if (!collective || !transform.activate(collective.lease.token) ||
      !transform.complete(collective.lease.token, true)) {
    return 54;
  }
  const std::array intermediate_keys{intermediate_outputs[0].key,
                                     intermediate_outputs[1].key};
  if (transform.begin_writeback(intermediate_keys, transform_intermediate,
                                2u) ||
      transform.drain_dirty(transform_intermediate, 2u)) {
    return 55;
  }
  const AuthorityResult retire_intermediate =
      transform.begin_discard(intermediate_keys, transform_intermediate, 2u);
  if (!retire_intermediate ||
      !transform.discard(retire_intermediate.lease.token)) {
    return 56;
  }
  const AuthorityResult retire_output =
      transform.begin_discard(transform_output_keys, transform_output, 2u);
  const std::array transform_regions{input_region, intermediate_region,
                                     output_region};
  if (!retire_output || !transform.discard(retire_output.lease.token) ||
      !transform.release_frames(transform_regions)) {
    return 57;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency
