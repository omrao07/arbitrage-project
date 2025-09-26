// include/shared_memory.hpp
#pragma once
// -----------------------------------------------------------------------------
// POSIX shared memory helpers (shm_open + mmap)
// -----------------------------------------------------------------------------
// Usage:
//   shm::MappedRegion reg = shm::MappedRegion::open_or_create("/risk_wall", 1<<20);
//   std::memset(reg.addr(), 0, reg.size());
//
// Notes:
//   • Name must start with '/' per POSIX (e.g., "/risk_wall").
//   • On Linux, objects appear under /dev/shm/<name-without-leading-slash>.
//   • Files are binary regions; do NOT commit them to git.
// -----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <system_error>

#include <sys/mman.h>   // mmap, munmap, PROT_*, MAP_*
#include <sys/stat.h>   // fstat, mode constants
#include <fcntl.h>      // shm_open, O_*
#include <unistd.h>     // ftruncate, close
#include <errno.h>
#include <string.h>     // ::strerror

namespace shm {

inline std::string normalize_name(std::string name) {
    if (name.empty() || name[0] != '/') name.insert(name.begin(), '/');
    // POSIX requires: a leading slash and no other slashes in the name.
    // We'll keep it lenient, but callers should prefer "/foo" style.
    return name;
}

enum class Access {
    ReadWrite,
    ReadOnly
};

class MappedRegion {
public:
    MappedRegion() = default;

    static MappedRegion open_or_create(const std::string& raw_name,
                                       std::size_t bytes,
                                       Access access = Access::ReadWrite,
                                       mode_t perms = 0600) {
        std::string name = normalize_name(raw_name);
        int oflag = (access == Access::ReadWrite) ? (O_CREAT | O_RDWR) : O_RDONLY;
        int fd = ::shm_open(name.c_str(), oflag, perms);
        if (fd < 0) throw_sys("shm_open(create)", name);

        // If writable, size it
        if (access == Access::ReadWrite) {
            if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
                int e = errno; ::close(fd);
                throw std::system_error(e, std::generic_category(), "ftruncate");
            }
        }

        int prot = (access == Access::ReadWrite) ? (PROT_READ | PROT_WRITE) : PROT_READ;
        void* addr = ::mmap(nullptr, bytes, prot, MAP_SHARED, fd, 0);
        if (addr == MAP_FAILED) {
            int e = errno; ::close(fd);
            throw std::system_error(e, std::generic_category(), "mmap");
        }
        return MappedRegion(name, fd, addr, bytes, access);
    }

    static MappedRegion open_existing(const std::string& raw_name,
                                      std::size_t bytes,
                                      Access access = Access::ReadOnly) {
        std::string name = normalize_name(raw_name);
        int oflag = (access == Access::ReadWrite) ? O_RDWR : O_RDONLY;
        int fd = ::shm_open(name.c_str(), oflag, 0);
        if (fd < 0) throw_sys("shm_open(open)", name);

        int prot = (access == Access::ReadWrite) ? (PROT_READ | PROT_WRITE) : PROT_READ;
        void* addr = ::mmap(nullptr, bytes, prot, MAP_SHARED, fd, 0);
        if (addr == MAP_FAILED) {
            int e = errno; ::close(fd);
            throw std::system_error(e, std::generic_category(), "mmap");
        }
        return MappedRegion(name, fd, addr, bytes, access);
    }

    // Unlink (remove) a named shared memory object. Safe to call even if not mapped here.
    static void unlink(const std::string& raw_name) {
        std::string name = normalize_name(raw_name);
        if (::shm_unlink(name.c_str()) != 0) {
            // If it doesn't exist, ignore; otherwise, surface error.
            if (errno != ENOENT) throw_sys("shm_unlink", name);
        }
    }

    // Move-only
    MappedRegion(MappedRegion&& other) noexcept { move_from(std::move(other)); }
    MappedRegion& operator=(MappedRegion&& other) noexcept {
        if (this != &other) { close(); move_from(std::move(other)); }
        return *this;
    }

    // No copy
    MappedRegion(const MappedRegion&) = delete;
    MappedRegion& operator=(const MappedRegion&) = delete;

    ~MappedRegion() { close(); }

    void* addr() const noexcept { return addr_; }
    std::size_t size() const noexcept { return size_; }
    Access access() const noexcept { return acc_; }
    int fd() const noexcept { return fd_; }
    const std::string& name() const noexcept { return name_; }

    // Explicit unmap/close (also happens in destructor)
    void close() noexcept {
        if (addr_ && size_) {
            ::munmap(addr_, size_);
            addr_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        size_ = 0;
        acc_ = Access::ReadOnly;
        name_.clear();
    }

private:
    std::string name_{};
    int         fd_{-1};
    void*       addr_{nullptr};
    std::size_t size_{0};
    Access      acc_{Access::ReadOnly};

    MappedRegion(const std::string& nm, int fd, void* addr, std::size_t sz, Access a)
        : name_(nm), fd_(fd), addr_(addr), size_(sz), acc_(a) {}

    static void throw_sys(const char* what, const std::string& name) {
        throw std::system_error(errno, std::generic_category(),
                                std::string(what) + " failed for '" + name + "'");
    }

    void move_from(MappedRegion&& o) noexcept {
        name_ = std::move(o.name_);
        fd_   = o.fd_;    o.fd_ = -1;
        addr_ = o.addr_;  o.addr_ = nullptr;
        size_ = o.size_;  o.size_ = 0;
        acc_  = o.acc_;
    }
};

} // namespace shm