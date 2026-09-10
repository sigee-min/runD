#include "../../../../../src/compute/buffer/state.hpp"
#include "local.hpp"

#include "../../../target/selection.hpp"
#include "src/accel/context/capability.hpp"
#include "src/compute/backend.hpp"
#include "src/compute/buffer/local.hpp"
#include "src/compute/device/state.hpp"
#include "src/compute/type.hpp"

#include <rund/compute/pipeline.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>

namespace rund_node_test_pipeline_residency {
namespace {

using rund::compute::Backend;
using rund::compute::Device;
using rund::compute::MemoryStats;
using rund::compute::Reason;
using rund::compute::Status;
using rund::compute::WriteStats;
using rund::compute::detail::accel_buffer;
using rund::compute::detail::AccelBufferState;
using rund::compute::detail::BufferState;
using rund::compute::detail::cpu_buffer;
using rund::compute::detail::CpuBufferState;
using rund::compute::detail::DeviceAccess;
using rund::compute::detail::DeviceState;
using rund::compute::detail::DownloadResult;
using rund::compute::detail::HostView;
using rund::compute::detail::make_buffer;
using rund::compute::detail::make_physical_buffer_view;
using rund::compute::detail::same_physical_buffer;
using rund::compute::detail::Type;
using rund::compute::detail::type_bytes;
using rund::compute::detail::UploadResult;
using rund::compute::detail::write_buffer_measured;

constexpr std::size_t WideCount = 8u;
constexpr std::size_t NarrowCount = 16u;
constexpr std::size_t ByteCount = WideCount * sizeof(std::uint64_t);

[[nodiscard]] bool native_descriptor(const BufferState &buffer, const Type type,
                                     const std::size_t count) noexcept {
  const AccelBufferState *const storage = accel_buffer(buffer);
  if (storage == nullptr || !storage->buffer) {
    return false;
  }
  const rund::AccelBuffer &native = storage->buffer;
  const std::uint64_t width = type_bytes(type);
  if (width == 0u || native.context_id == 0u ||
      native.byte_extent != ByteCount || native.scalar_width_bytes != width ||
      native.count != count || !native.buffer.check.ok ||
      native.buffer.id == 0u || native.buffer.bytes != ByteCount ||
      native.buffer.storage_bytes < ByteCount ||
      native.buffer.owner == nullptr || native.buffer.handle == nullptr ||
      native.owner == nullptr || native.handle == nullptr) {
    return false;
  }
  const rund::kernel::ResidentBufferRef &resident = native.resident;
  return resident.id == native.buffer.id &&
         resident.bytes == native.buffer.bytes && resident.offset_bytes == 0u &&
         resident.element_bytes == width && resident.stride_bytes == width &&
         resident.count == count;
}

[[nodiscard]] bool same_native_storage(const BufferState &left,
                                       const BufferState &right) noexcept {
  const AccelBufferState *const lhs = accel_buffer(left);
  const AccelBufferState *const rhs = accel_buffer(right);
  if (lhs == nullptr || rhs == nullptr || !lhs->buffer || !rhs->buffer) {
    return false;
  }
  const auto lhs_token =
      rund::node::accel::detail::LookupAccelBufferToken(lhs->buffer.handle);
  const auto rhs_token =
      rund::node::accel::detail::LookupAccelBufferToken(rhs->buffer.handle);
  return lhs_token != nullptr && rhs_token != nullptr &&
         lhs_token->backend_handle == rhs_token->backend_handle &&
         lhs->buffer.buffer.id == rhs->buffer.buffer.id &&
         lhs->buffer.buffer.bytes == rhs->buffer.buffer.bytes &&
         lhs->buffer.buffer.storage_bytes == rhs->buffer.buffer.storage_bytes &&
         lhs->buffer.resident.id == rhs->buffer.resident.id &&
         lhs->buffer.resident.bytes == rhs->buffer.resident.bytes;
}

[[nodiscard]] bool write_exact(const std::shared_ptr<BufferState> &buffer,
                               const void *const data, const std::size_t count,
                               const Type type) noexcept {
  WriteStats stats{};
  UploadResult transfer{};
  const Status written = write_buffer_measured(
      buffer, HostView{.data = data, .count = count, .type = type}, stats,
      transfer);
  if (!written || !transfer.status) {
    std::fprintf(stderr, "physical view write reason=%u transfer=%u\n",
                 static_cast<unsigned>(written.reason()),
                 static_cast<unsigned>(transfer.status.reason()));
  }
  return written && transfer.status;
}

[[nodiscard]] bool read_exact(DeviceState &device,
                              const std::shared_ptr<BufferState> &buffer,
                              void *const data,
                              const std::size_t bytes) noexcept {
  if (buffer == nullptr || data == nullptr || buffer->bytes != bytes) {
    return false;
  }
  if (device.backend == Backend::Cpu) {
    const CpuBufferState *const storage = cpu_buffer(*buffer);
    if (storage == nullptr || storage->data == nullptr ||
        storage->bytes != bytes) {
      return false;
    }
    std::memcpy(data, storage->data.get(), bytes);
    return true;
  }
  if (device.ops == nullptr || device.ops->download == nullptr) {
    return false;
  }
  const DownloadResult downloaded =
      device.ops->download(device, *buffer, data, bytes, 0u);
  if (!downloaded.status || !downloaded.payload_hash_valid) {
    std::fprintf(
        stderr,
        "physical view read reason=%u hash=%d confirmed=%llu expected=%zu\n",
        static_cast<unsigned>(downloaded.status.reason()),
        downloaded.payload_hash_valid,
        static_cast<unsigned long long>(downloaded.confirmed_bytes), bytes);
  }
  // The single-buffer native transfer reports status/hash, not the batch
  // receipt's confirmed-byte prefix. The caller compares every returned byte.
  return downloaded.status && downloaded.payload_hash_valid;
}

[[nodiscard]] int check_backend(const Backend backend) {
  auto opened =
      rund::compute::open(rund::node::test_contract::target_for(backend, 1u));
  if (!opened) {
    return opened.reason() == Reason::AdapterUnavailable ? 0 : 1;
  }
  Device device = std::move(opened).value();
  const std::shared_ptr<DeviceState> state = DeviceAccess::state(device);
  if (state == nullptr || state->backend != backend) {
    return 2;
  }

  auto root_result = make_buffer(state, Type::U64, WideCount);
  if (!root_result) {
    return 3;
  }
  const std::shared_ptr<BufferState> root = std::move(root_result).value();
  if (root == nullptr || root->type != Type::U64 || root->count != WideCount ||
      root->bytes != ByteCount || root->physical_bytes < root->bytes ||
      !root->memory_accounted || root->physical_owner != nullptr) {
    return 4;
  }

  const MemoryStats accounted = device.memory();
  for (const std::size_t invalid_count :
       {NarrowCount - 1u, NarrowCount + 1u,
        std::numeric_limits<std::size_t>::max()}) {
    const auto rejected =
        make_physical_buffer_view(root, Type::U32, invalid_count);
    if (rejected || rejected.reason() != Reason::BufferCapacity ||
        device.memory() != accounted) {
      return 12;
    }
  }
  auto narrow_result = make_physical_buffer_view(root, Type::U32, NarrowCount);
  if (!narrow_result) {
    return 5;
  }
  const std::shared_ptr<BufferState> narrow = std::move(narrow_result).value();
  auto wide_result = make_physical_buffer_view(narrow, Type::U64, WideCount);
  if (!wide_result) {
    return 6;
  }
  const std::shared_ptr<BufferState> wide = std::move(wide_result).value();
  if (narrow == nullptr || wide == nullptr || narrow->device != state ||
      wide->device != state || narrow->type != Type::U32 ||
      narrow->count != NarrowCount || narrow->bytes != ByteCount ||
      wide->type != Type::U64 || wide->count != WideCount ||
      wide->bytes != ByteCount ||
      narrow->physical_bytes != root->physical_bytes ||
      wide->physical_bytes != root->physical_bytes ||
      narrow->memory_accounted || wide->memory_accounted ||
      narrow->physical_owner != root || wide->physical_owner != root ||
      !same_physical_buffer(*root, *narrow) ||
      !same_physical_buffer(*root, *wide) || device.memory() != accounted) {
    return 7;
  }

  if (backend == Backend::Cpu) {
    const CpuBufferState *const root_storage = cpu_buffer(*root);
    const CpuBufferState *const narrow_storage = cpu_buffer(*narrow);
    const CpuBufferState *const wide_storage = cpu_buffer(*wide);
    if (root_storage == nullptr || narrow_storage == nullptr ||
        wide_storage == nullptr || root_storage->data == nullptr ||
        root_storage->data != narrow_storage->data ||
        root_storage->data != wide_storage->data ||
        root_storage->bytes != ByteCount ||
        narrow_storage->bytes != ByteCount ||
        wide_storage->bytes != ByteCount) {
      return 8;
    }
  } else if (!native_descriptor(*root, Type::U64, WideCount) ||
             !native_descriptor(*narrow, Type::U32, NarrowCount) ||
             !native_descriptor(*wide, Type::U64, WideCount) ||
             !same_native_storage(*root, *narrow) ||
             !same_native_storage(*root, *wide)) {
    return 9;
  }

  constexpr std::array<std::uint32_t, NarrowCount> narrow_values{
      0x10203040u, 0x50607080u, 0x90a0b0c0u, 0xd0e0f001u,
      0x12345678u, 0x9abcdef0u, 0x0badc0deu, 0xfeedbeefu,
      0x01020304u, 0x05060708u, 0x11121314u, 0x15161718u,
      0x21222324u, 0x25262728u, 0x31323334u, 0x35363738u};
  constexpr std::array<std::uint64_t, WideCount> wide_values{
      0x0102030405060708ull, 0x1112131415161718ull, 0x2122232425262728ull,
      0x3132333435363738ull, 0x4142434445464748ull, 0x5152535455565758ull,
      0x6162636465666768ull, 0x7172737475767778ull};
  std::array<std::byte, ByteCount> expected{};
  std::array<std::byte, ByteCount> observed{};

  std::memcpy(expected.data(), narrow_values.data(), expected.size());
  if (!write_exact(narrow, narrow_values.data(), narrow_values.size(),
                   Type::U32) ||
      !read_exact(*state, wide, observed.data(), observed.size()) ||
      std::memcmp(expected.data(), observed.data(), expected.size()) != 0) {
    return 10;
  }
  std::memcpy(expected.data(), wide_values.data(), expected.size());
  if (!write_exact(wide, wide_values.data(), wide_values.size(), Type::U64) ||
      !read_exact(*state, narrow, observed.data(), observed.size()) ||
      std::memcmp(expected.data(), observed.data(), expected.size()) != 0) {
    return 11;
  }
  return 0;
}

} // namespace

int CheckPhysicalBufferViews() {
  for (const Backend backend :
       rund::node::test_contract::selected_compute_backends()) {
    if (const int result = check_backend(backend); result != 0) {
      std::fprintf(stderr, "physical view backend=%u result=%d\n",
                   static_cast<unsigned>(backend), result);
      return result;
    }
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency
