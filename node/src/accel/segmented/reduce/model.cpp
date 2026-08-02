#include "model.hpp"

namespace rund::node::accel::detail {

void AppendSegmentedReduceShaderModel(std::string &source) {
  backend_source_recipe::StringSink sink{source};
  (void)AppendSegmentedReduceShaderModel(sink);
}

} // namespace rund::node::accel::detail
