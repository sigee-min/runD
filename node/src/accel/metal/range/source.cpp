#include "../../kernel/backend/source_recipe.hpp"
#include "local.hpp"
#include "source/build.hpp"

#include <string>

namespace rund::node::accel::detail {

std::string MetalRangeSource(const RangeExec &execution) {
  const auto emit = [&execution](auto &sink) noexcept(
                        noexcept(EmitMetalRangeSource(sink, execution))) {
    return EmitMetalRangeSource(sink, execution);
  };
  return backend_source_recipe::materialize(emit);
}

bool MetalRangeSourceUpperBytes(const RangeExec &execution,
                                std::uint64_t &upper) noexcept {
  const auto emit =
      [&execution](backend_source_recipe::CountSink &sink) noexcept {
        return EmitMetalRangeSource(sink, execution);
      };
  return backend_source_recipe::bytes(emit, upper);
}

} // namespace rund::node::accel::detail
