#include "local.hpp"

#include <array>

namespace rund_node_test_pipeline {

[[nodiscard]] int CheckSurfaceAlias(
    rund::compute::Device &device,
    const rund::compute::Buffer<std::int32_t> &input) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 4u> input_values{1, 2, 3, 4};
  auto aliases_program =
      on(device)
          .map<std::int32_t>("pipeline-alias-source", input_values.size(),
                             [](auto value) { return value * 3; })
          .branch([](auto source) {
            auto next = source.map("pipeline-alias-next",
                                   [](auto value) { return value + 1; });
            return outputs(source, next, source);
          })
          .compile();
  auto alias_source = device.buffer<std::int32_t>(input_values.size());
  auto alias_next = device.buffer<std::int32_t>(input_values.size());
  if (!aliases_program || !alias_source || !alias_next) {
    return 8;
  }
  auto alias_pipeline =
      pipeline(device)
          .then(*aliases_program, read(input),
                write(*alias_source, *alias_next, *alias_source))
          .prepare();
  if (!alias_pipeline || !alias_pipeline->run()) {
    return 9;
  }
  std::array<std::int32_t, 4u> values{};
  std::array<std::int32_t, 4u> weights{};
  if (!ReadExact(*alias_pipeline, *alias_source, values) ||
      !ReadExact(*alias_pipeline, *alias_next, weights) ||
      values != std::array<std::int32_t, 4u>{3, 6, 9, 12} ||
      weights != std::array<std::int32_t, 4u>{4, 7, 10, 13} ||
      alias_pipeline->stats().pipeline.resource_count != 3u) {
    return 10;
  }

  auto bad_alias_source = device.buffer<std::int32_t>(input_values.size());
  if (!bad_alias_source) {
    return 11;
  }
  auto rejected_projection =
      pipeline(device)
          .then(*aliases_program, read(input),
                write(*alias_source, *alias_next, *bad_alias_source))
          .prepare();
  if (rejected_projection ||
      rejected_projection.reason() != Reason::BindingAliasUnsupported) {
    return 12;
  }
  return 0;
}

} // namespace rund_node_test_pipeline
