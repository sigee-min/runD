#include "../publication.hpp"

#include "../contract.hpp"

#include "../../../size.hpp"
#include "../../../type.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <variant>

namespace rund::compute::detail {
namespace {

[[nodiscard]] node::accel::detail::PreparedKernelPublicationViewIdentity
project_view(const PipelinePublicationViewIdentity &view) noexcept {
  return node::accel::detail::PreparedKernelPublicationViewIdentity{
      .backing_bytes = view.backing_bytes,
      .offset_bytes = view.offset_bytes,
      .count = view.count,
      .stride_bytes = view.stride_bytes,
      .element_bytes = view.element_bytes,
      .resource_ordinal = view.resource_ordinal,
      .usage = view.usage,
  };
}

} // namespace

bool mix_pipeline_publication_public_identity(
    PipelineHash &hash, const PipelinePublicationPlan &publication,
    const PipelineWindowControl &control) noexcept {
  static_assert(static_cast<std::uint8_t>(PipelinePublicationKind::Terminal) ==
                0u);
  static_assert(static_cast<std::uint8_t>(PipelinePublicationKind::Window) ==
                1u);

  const auto *window = std::get_if<PipelineWindowPublicationPlan>(&publication);
  const auto *terminal =
      std::get_if<PipelineTerminalPublicationPlan>(&publication);
  if ((window == nullptr && terminal == nullptr) ||
      control.final < PipelineWindow::first ||
      control.final > PipelineWindow::second) {
    return false;
  }
  const PipelinePublicationViewPlan &source =
      window != nullptr ? window->source : terminal->sources[control.final];
  const PipelinePublicationViewPlan &target =
      window != nullptr ? window->target.view : terminal->target.view;
  const PipelinePublicationViewIdentity &source_identity = source.identity;
  const PipelinePublicationViewIdentity &target_identity = target.identity;
  const PipelinePublicationViewPlan *const count =
      window != nullptr ? &control.count : nullptr;
  if (!valid_type(source.type) || !valid_format(source.type, source.format) ||
      source_identity.element_bytes == 0u ||
      target_identity.offset_bytes % source_identity.element_bytes != 0u ||
      target_identity.stride_bytes % source_identity.element_bytes != 0u ||
      (count != nullptr &&
       (count->identity.element_bytes == 0u ||
        count->identity.offset_bytes % count->identity.element_bytes != 0u))) {
    return false;
  }

  hash.number(source_identity.resource_ordinal);
  hash.number(target_identity.resource_ordinal);
  hash.number(static_cast<std::uint64_t>(source.type));
  hash.format(source.format);
  hash.number(source_identity.count);
  hash.number(target_identity.offset_bytes / source_identity.element_bytes);
  hash.number(target_identity.stride_bytes / source_identity.element_bytes);
  hash.number(source_identity.element_bytes);
  hash.number(static_cast<std::uint64_t>(window != nullptr ? window->state
                                                           : terminal->state) +
              1u);
  hash.number(window != nullptr ? window->output.value
                                : terminal->output.value);
  hash.byte(static_cast<std::uint8_t>(pipeline_publication_kind(publication)));
  hash.number(count == nullptr ? 0u
                               : count->identity.offset_bytes /
                                     count->identity.element_bytes);
  hash.number(window != nullptr ? control.maximum : 0u);
  hash.number(window != nullptr ? control.tile : 0u);
  return true;
}

node::accel::detail::PreparedKernelPublicationIdentity
project_pipeline_publication_identity(
    const PipelinePublicationPlan &publication,
    const PipelineWindowControl &control,
    const std::uint32_t outer_bound) noexcept {
  node::accel::detail::PreparedKernelPublicationIdentity identity{};
  if (const auto *terminal =
          std::get_if<PipelineTerminalPublicationPlan>(&publication)) {
    for (std::size_t bank = 0u; bank < terminal->sources.size(); ++bank) {
      identity.sources[bank] = project_view(terminal->sources[bank].identity);
    }
    identity.target = project_view(terminal->target.view.identity);
    identity.state = terminal->state;
    identity.final = control.final;
    identity.kind =
        node::accel::detail::PreparedKernelPublicationKind::Terminal;
    return identity;
  }
  const PipelineWindowPublicationPlan &window =
      std::get<PipelineWindowPublicationPlan>(publication);
  const auto source = project_view(window.source.identity);
  std::fill(std::begin(identity.sources), std::end(identity.sources), source);
  identity.count = project_view(control.count.identity);
  identity.target = project_view(window.target.view.identity);
  identity.state = window.state;
  identity.maximum = control.maximum;
  identity.tile = control.tile;
  identity.outer_bound = outer_bound;
  identity.kind = node::accel::detail::PreparedKernelPublicationKind::Window;
  return identity;
}

} // namespace rund::compute::detail
