#include "local.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include <utility>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace rund::node::replay_detail::payload {
namespace {

constexpr std::string_view kGenerationPrefix = ".rund-replay-spill-v1-";
constexpr std::string_view kGenerationMarker = ".rund-owner";
constexpr std::string_view kGenerationLease = ".rund-lease";
constexpr std::string_view kGenerationMarkerContents =
    "runD replay spill generation v1\n";
std::atomic<std::uint64_t> gGenerationSequence{1u};

[[nodiscard]] bool has_generation_prefix(const std::filesystem::path &path) {
  return path.filename().string().starts_with(kGenerationPrefix);
}

[[nodiscard]] bool marker_matches(const std::filesystem::path &directory) {
  std::ifstream marker{directory / kGenerationMarker, std::ios::binary};
  if (!marker) {
    return false;
  }
  std::array<char, kGenerationMarkerContents.size() + 1u> contents{};
  marker.read(contents.data(), static_cast<std::streamsize>(contents.size()));
  return marker.gcount() ==
             static_cast<std::streamsize>(kGenerationMarkerContents.size()) &&
         marker.eof() &&
         std::string_view{contents.data(), kGenerationMarkerContents.size()} ==
             kGenerationMarkerContents;
}

void close_descriptor(const int descriptor) noexcept {
  if (descriptor >= 0) {
    static_cast<void>(::close(descriptor));
  }
}

[[nodiscard]] bool write_all(const int descriptor,
                             const std::string_view bytes) noexcept {
  std::size_t offset = 0u;
  while (offset < bytes.size()) {
    const ssize_t written =
        ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
    if (written > 0) {
      offset += static_cast<std::size_t>(written);
      continue;
    }
    if (written < 0 && errno == EINTR) {
      continue;
    }
    return false;
  }
  return true;
}

void remove_owned(const std::filesystem::path &root,
                  const std::filesystem::path &directory) noexcept {
  try {
    if (directory.parent_path() != root || !has_generation_prefix(directory) ||
        !marker_matches(directory)) {
      return;
    }
    std::error_code error{};
    std::filesystem::remove_all(directory, error);
  } catch (...) {
  }
}

void scavenge(const std::filesystem::path &root) noexcept {
  try {
    std::error_code error{};
    std::filesystem::directory_iterator entries{root, error};
    if (error) {
      return;
    }
    for (const std::filesystem::directory_entry &entry : entries) {
      const std::filesystem::path directory = entry.path();
      if (!entry.is_directory(error) || error ||
          !has_generation_prefix(directory) || !marker_matches(directory)) {
        error.clear();
        continue;
      }
      const std::filesystem::path lease_path = directory / kGenerationLease;
      const int lease = ::open(lease_path.c_str(), O_RDWR | O_CLOEXEC,
                               static_cast<mode_t>(0));
      if (lease < 0) {
        continue;
      }
      if (::flock(lease, LOCK_EX | LOCK_NB) == 0) {
        remove_owned(root, directory);
        static_cast<void>(::flock(lease, LOCK_UN));
      }
      close_descriptor(lease);
    }
  } catch (...) {
  }
}

void scavenge_once(const std::filesystem::path &root) noexcept {
  static std::mutex gate{};
  static std::unordered_set<std::string> scanned_roots{};
  try {
    std::lock_guard lock{gate};
    if (scanned_roots.insert(root.string()).second) {
      scavenge(root);
    }
  } catch (...) {
    // Recovery is best-effort; exclusive generation creation remains safe.
  }
}

[[nodiscard]] std::string generation_name() {
  const std::uint64_t sequence =
      gGenerationSequence.fetch_add(1u, std::memory_order_relaxed);
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  std::ostringstream name{};
  name << kGenerationPrefix << std::hex
       << static_cast<std::uint64_t>(
              static_cast<unsigned long long>(::getpid()))
       << '-' << static_cast<std::uint64_t>(now) << '-' << sequence;
  return name.str();
}

} // namespace

std::shared_ptr<const SpillGeneration>
SpillGeneration::Create(const std::string &requested_root,
                        ::rund::storage::Budget budget,
                        const std::size_t reservation_capacity) noexcept {
  try {
    if (!budget) {
      return {};
    }
    std::vector<::rund::storage::Reservation> reservations{};
    reservations.reserve(reservation_capacity);
    std::error_code error{};
    std::filesystem::path root =
        std::filesystem::absolute(std::filesystem::path{requested_root}, error);
    if (error) {
      return {};
    }
    root = root.lexically_normal();
    std::filesystem::create_directories(root, error);
    if (error) {
      return {};
    }
    scavenge_once(root);

    constexpr std::size_t kAttempts = 128u;
    for (std::size_t attempt = 0u; attempt < kAttempts; ++attempt) {
      const std::filesystem::path directory = root / generation_name();
      error.clear();
      if (!std::filesystem::create_directory(directory, error)) {
        if (error) {
          return {};
        }
        continue;
      }

      const std::filesystem::path lease_path = directory / kGenerationLease;
      const int lease =
          ::open(lease_path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC,
                 static_cast<mode_t>(0600));
      if (lease < 0 || ::flock(lease, LOCK_EX | LOCK_NB) != 0) {
        close_descriptor(lease);
        std::filesystem::remove_all(directory, error);
        return {};
      }

      const std::filesystem::path marker_path = directory / kGenerationMarker;
      const int marker =
          ::open(marker_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                 static_cast<mode_t>(0600));
      const bool marked = marker >= 0 &&
                          write_all(marker, kGenerationMarkerContents) &&
                          ::fsync(marker) == 0;
      close_descriptor(marker);
      if (!marked) {
        std::filesystem::remove_all(directory, error);
        static_cast<void>(::flock(lease, LOCK_UN));
        close_descriptor(lease);
        return {};
      }

      return std::shared_ptr<const SpillGeneration>{
          new SpillGeneration{root.string(), directory.string(), lease,
                              std::move(budget), std::move(reservations)}};
    }
  } catch (...) {
  }
  return {};
}

SpillGeneration::~SpillGeneration() {
  const std::filesystem::path root{root_};
  const std::filesystem::path directory{directory_};
  remove_owned(root, directory);
  if (lease_descriptor_ >= 0) {
    static_cast<void>(::flock(lease_descriptor_, LOCK_UN));
    close_descriptor(lease_descriptor_);
  }
}

::rund::storage::Reservation
SpillGeneration::Reserve(const std::uint64_t allocated_bytes) const noexcept {
  return budget_.reserve(allocated_bytes);
}

bool SpillGeneration::Stage(
    ::rund::storage::Reservation reservation) const noexcept {
  if (!reservation || reservations_.size() >= reservations_.capacity() ||
      !rund::kernel::checked::add(reserved_bytes_,
                                  reservation.max_allocated_bytes())) {
    return false;
  }
  reserved_bytes_ += reservation.max_allocated_bytes();
  reservations_.push_back(std::move(reservation));
  return true;
}

bool SpillGeneration::CommitLast(
    const ::rund::storage::Usage usage) const noexcept {
  if (reservations_.empty() || reservations_.back().committed() ||
      !rund::kernel::checked::add(usage_.physical_bytes,
                                  usage.physical_bytes) ||
      !rund::kernel::checked::add(usage_.allocated_bytes,
                                  usage.allocated_bytes)) {
    return false;
  }
  const std::uint64_t reserved = reservations_.back().max_allocated_bytes();
  if (!reservations_.back().commit(usage)) {
    return false;
  }
  reserved_bytes_ -= reserved;
  usage_.physical_bytes += usage.physical_bytes;
  usage_.allocated_bytes += usage.allocated_bytes;
  return true;
}

void SpillGeneration::Rollback(
    const std::size_t reservation_count) const noexcept {
  while (reservations_.size() > reservation_count) {
    const ::rund::storage::Reservation &reservation = reservations_.back();
    if (reservation.committed()) {
      const ::rund::storage::Usage released = reservation.usage();
      usage_.physical_bytes -= released.physical_bytes;
      usage_.allocated_bytes -= released.allocated_bytes;
    } else {
      reserved_bytes_ -= reservation.max_allocated_bytes();
    }
    reservations_.pop_back();
  }
}

std::size_t SpillGeneration::reservation_count() const noexcept {
  return reservations_.size();
}

::rund::storage::Usage SpillGeneration::usage() const noexcept {
  return usage_;
}

std::uint64_t SpillGeneration::reserved_bytes() const noexcept {
  return reserved_bytes_;
}

} // namespace rund::node::replay_detail::payload
