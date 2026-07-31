#pragma once

namespace rund::compute {
class Device;
}

namespace node_compute_cache_contract {

int RunService();
int RunAsync(rund::compute::Device &device);
int RunCapacity();
int RunLifetime();
int RunFailure();

} // namespace node_compute_cache_contract
