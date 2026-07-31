#include "../resident/state.hpp"
#include "../state.hpp"
#include "../pipeline/artifact/index.hpp"

#include <memory>

namespace rund::node::accel::detail {

MetalAdapter::MetalAdapter()
    : pipeline_index(std::make_unique<MetalPipelineIndex>()),
      resident(std::make_unique<MetalResidentState>()) {}

MetalAdapter::~MetalAdapter() = default;

} // namespace rund::node::accel::detail
