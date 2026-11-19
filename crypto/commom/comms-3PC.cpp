#include <crypto/commom/comms-3PC.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;
using Deadline = Clock::time_point;
using Bytes = std::vector<std::uint8_t>;

[[noreturn]] void systemFailure(const char* operation, int error = errno)
{
    throw std::system_error(error, std::generic_category(), operation);
}

[[noreturn]] void timedOut()
{
    throw std::system_error(std::make_error_code(std::errc::timed_out), "RSS TCP operation");
}

Deadline deadlineAfter(std::chrono::milliseconds timeout)
{
    const auto now = Clock::now();
    if (timeout.count() <= 0 ||
        timeout > std::chrono::duration_cast<std::chrono::milliseconds>(Deadline::max() - now)) {
        throw std::invalid_argument("RSS timeout must be positive and representable");
    }
    return now + timeout;
}

int remainingMs(Deadline deadline)
{
    const auto now = Clock::now();
    if (now >= deadline) timedOut();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
    return static_cast<int>(std::min<std::int64_t>(ms + 1, INT_MAX));
}

void pollUntil(pollfd* descriptors, nfds_t count, Deadline deadline)
{
    for (;;) {
        const int result = ::poll(descriptors, count, remainingMs(deadline));
        if (result > 0) {
            for (nfds_t i = 0; i < count; ++i) {
                if (descriptors[i].revents & POLLNVAL) systemFailure("RSS poll", EBADF);
            }
            return;
        }
        if (result == 0) continue; // Recheck the absolute deadline.
        if (errno != EINTR) systemFailure("RSS poll");
    }
}

void retryPause(Deadline deadline)
{
    const int delay = std::min(20, remainingMs(deadline));
    if (::poll(nullptr, 0, delay) < 0 && errno != EINTR) systemFailure("RSS connect retry");
}

class OwnedFd {
public:
    explicit OwnedFd(int fd = -1) : fd_(fd) {}
    ~OwnedFd() { if (fd_ >= 0) ::close(fd_); }
    OwnedFd(const OwnedFd&) = delete;
    OwnedFd& operator=(const OwnedFd&) = delete;
    int get() const { return fd_; }
    int release() { const int fd = fd_; fd_ = -1; return fd; }
private:
    int fd_;
};

int partyIndex(RssParty party)
{
    const int index = static_cast<int>(party);
    if (index < 0 || index > 2) throw std::invalid_argument("RSS party must be P0, P1 or P2");
    return index;
}

std::size_t elementBytes(int bw)
{
    if (bw < 1 || bw > 64) throw std::invalid_argument("RSS bit width must be in [1, 64]");
    return bw <= 8 ? 1 : bw <= 16 ? 2 : bw <= 32 ? 4 : 8;
}

std::size_t checkedBytes(const void* data, std::size_t count, std::size_t unit)
{
    if (count > std::numeric_limits<std::size_t>::max() / unit) {
        throw std::invalid_argument("RSS array byte count overflow");
    }
    if (count != 0 && data == nullptr) throw std::invalid_argument("RSS nonempty buffer is null");
    return count * unit;
}

void checkRawOverlap(const void* sendData, std::size_t sendBytes,
                     const void* recvData, std::size_t recvBytes)
{
    if (!sendBytes || !recvBytes) return;
    const auto s = reinterpret_cast<std::uintptr_t>(sendData);
    const auto r = reinterpret_cast<std::uintptr_t>(recvData);
    if ((s >= r && s - r < recvBytes) || (r >= s && r - s < sendBytes)) {
        throw std::invalid_argument("RSS raw exchange buffers overlap");
    }
}

GroupElement normalized(GroupElement value, int bw)
{
    mod(value, bw); // All callers validated bw before any I/O.
    return value;
}

void encodeElement(std::uint8_t* out, GroupElement value, std::size_t width, int bw)
{
    value = normalized(value, bw);
    for (std::size_t j = 0; j < width; ++j) out[j] = static_cast<std::uint8_t>(value >> (8 * j));
}

GroupElement decodeElement(const std::uint8_t* in, std::size_t width, int bw)
{
    GroupElement value = 0;
    for (std::size_t j = 0; j < width; ++j) value |= GroupElement{in[j]} << (8 * j);
    return normalized(value, bw);
}

Bytes encodeElements(const GroupElement* data, std::size_t count, int bw)
{
    const auto width = elementBytes(bw);
    checkedBytes(data, count, sizeof(GroupElement));
    Bytes wire(checkedBytes(data, count, width));
    for (std::size_t i = 0; i < count; ++i) encodeElement(wire.data() + i * width, data[i], width, bw);
    return wire;
}

void decodeElements(const Bytes& wire, GroupElement* data, std::size_t count, int bw)
{
    const auto width = elementBytes(bw);
    for (std::size_t i = 0; i < count; ++i) data[i] = decodeElement(wire.data() + i * width, width, bw);
}

template<class Share>
Bytes encodeShares(const Share* data, std::size_t count, int bw)
{
    const auto width = elementBytes(bw);
    checkedBytes(data, count, sizeof(Share));
    Bytes wire(checkedBytes(data, count, 2 * width));
    for (std::size_t i = 0; i < count; ++i) {
        encodeElement(wire.data() + i * 2 * width, data[i].first, width, bw);
        encodeElement(wire.data() + (i * 2 + 1) * width, data[i].second, width, bw);
    }
    return wire;
}

template<class Share>
void decodeShares(const Bytes& wire, Share* data, std::size_t count, int bw)
{
    const auto width = elementBytes(bw);
    for (std::size_t i = 0; i < count; ++i) {
        data[i].first = decodeElement(wire.data() + i * 2 * width, width, bw);
        data[i].second = decodeElement(wire.data() + (i * 2 + 1) * width, width, bw);
    }
}

void makeNonblocking(int fd)
{
    int flags;
    do { flags = ::fcntl(fd, F_GETFL); } while (flags < 0 && errno == EINTR);
    if (flags < 0) systemFailure("RSS fcntl F_GETFL");
    int result;
    do { result = ::fcntl(fd, F_SETFL, flags | O_NONBLOCK); } while (result < 0 && errno == EINTR);
    if (result < 0) systemFailure("RSS fcntl F_SETFL");
    do { flags = ::fcntl(fd, F_GETFD); } while (flags < 0 && errno == EINTR);
    if (flags < 0) systemFailure("RSS fcntl F_GETFD");
    do { result = ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC); } while (result < 0 && errno == EINTR);
    if (result < 0) systemFailure("RSS fcntl F_SETFD");
}

void noDelay(int fd)
{
    const int one = 1;
    if (::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) < 0) {
        systemFailure("RSS TCP_NODELAY");
    }
}

int tcpSocket()
{
    const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) systemFailure("RSS socket");
    return fd;
}

bool retryableConnect(int error)
{
    return error == ECONNREFUSED || error == ETIMEDOUT || error == ECONNRESET ||
           error == ENETUNREACH || error == EHOSTUNREACH;
}

int connectUntil(const sockaddr_in& address, const sockaddr_in& local, Deadline deadline)
{
    for (;;) {
        remainingMs(deadline);
        OwnedFd socket(tcpSocket());
        noDelay(socket.get());
        // WSL can route 127.0.0.1 through a host forwarding interface. Selecting a
        // different loopback destination alone is insufficient: the default source
        // may still be 127.0.0.1, sending replies through that interface.
        // Use the initiating party's configured loopback address with an ephemeral
        // port. For non-loopback links, keep the operating system's source selection.
        if ((ntohl(address.sin_addr.s_addr) >> 24) == 127 &&
            (ntohl(local.sin_addr.s_addr) >> 24) == 127) {
            sockaddr_in source = local;
            source.sin_port = 0;
            if (::bind(socket.get(), reinterpret_cast<const sockaddr*>(&source), sizeof(source)) < 0) {
                systemFailure("RSS bind loopback source");
            }
        }
        int error = 0;
        if (::connect(socket.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) < 0) {
            error = errno;
            if (error == EINPROGRESS || error == EINTR || error == EALREADY) {
                pollfd event{socket.get(), POLLOUT, 0};
                // A stalled SYN must not consume the entire startup deadline.
                const auto attempt = std::min(deadline, Clock::now() + std::chrono::seconds{1});
                try {
                    pollUntil(&event, 1, attempt);
                    socklen_t length = sizeof(error);
                    if (::getsockopt(socket.get(), SOL_SOCKET, SO_ERROR, &error, &length) < 0) {
                        systemFailure("RSS connect SO_ERROR");
                    }
                } catch (const std::system_error& failure) {
                    if (failure.code() != std::errc::timed_out) throw;
                    error = ETIMEDOUT;
                }
            }
        }
        if (error == 0) return socket.release();
        if (!retryableConnect(error)) systemFailure("RSS connect", error);
        retryPause(deadline);
    }
}

int acceptUntil(int listener, Deadline deadline)
{
    for (;;) {
        remainingMs(deadline);
        const int fd = ::accept4(listener, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (fd >= 0) {
            OwnedFd accepted(fd);
            noDelay(fd);
            return accepted.release();
        }
        if (errno == EINTR || errno == ECONNABORTED) continue;
        if (errno != EAGAIN && errno != EWOULDBLOCK) systemFailure("RSS accept");
        pollfd event{listener, POLLIN, 0};
        pollUntil(&event, 1, deadline);
    }
}

std::array<std::uint8_t, 8> hello(int from, int to)
{
    return {{'R', 'S', 'S', '3', 1, 0, static_cast<std::uint8_t>(from),
             static_cast<std::uint8_t>(to)}};
}

int checkHello(const std::array<std::uint8_t, 8>& wire, int self)
{
    const int from = wire[6];
    if (from > 2 || from == self || wire != hello(from, self)) {
        throw std::runtime_error("RSS invalid protocol version or party handshake");
    }
    return from;
}

void systemRandom(void* output, std::size_t length)
{
    auto* bytes = static_cast<std::uint8_t*>(output);
    std::size_t done = 0;
    while (done < length) {
        // Fail explicitly if entropy is not ready, instead of an unbounded wait.
        const auto n = ::getrandom(bytes + done, length - done, GRND_NONBLOCK);
        if (n > 0) done += static_cast<std::size_t>(n);
        else if (n < 0 && errno == EINTR) continue;
        else systemFailure("RSS getrandom", n == 0 ? EIO : errno);
    }
}

} // namespace

Peer3PC::Peer3PC(int connectedSocket, std::chrono::milliseconds ioTimeout)
    : socket_(connectedSocket), timeout_(ioTimeout)
{
    try {
        if (socket_ < 0) throw std::invalid_argument("RSS socket is invalid");
        deadlineAfter(timeout_);
        makeNonblocking(socket_);
    } catch (...) {
        close();
        throw;
    }
}

Peer3PC::~Peer3PC() { close(); }

void Peer3PC::requireOpen() const
{
    if (socket_ < 0) throw std::logic_error("RSS peer connection is closed");
}

void Peer3PC::close() noexcept
{
    if (socket_ >= 0) {
        ::shutdown(socket_, SHUT_RDWR);
        ::close(socket_);
        socket_ = -1;
    }
}

std::uint64_t Peer3PC::bytesSent() const noexcept { return sent_; }
std::uint64_t Peer3PC::bytesReceived() const noexcept { return received_; }
void Peer3PC::zeroBytesSent() noexcept { sent_ = 0; }
void Peer3PC::zeroBytesReceived() noexcept { received_ = 0; }

void Peer3PC::exchangeUntil(Peer3PC& sender, const void* sendData, std::size_t sendBytes,
                           Peer3PC& receiver, void* recvData, std::size_t recvBytes,
                           Deadline deadline)
{
    checkedBytes(sendData, sendBytes, 1);
    checkedBytes(recvData, recvBytes, 1);
    checkRawOverlap(sendData, sendBytes, recvData, recvBytes);
    sender.requireOpen();
    receiver.requireOpen();
    const auto* source = static_cast<const std::uint8_t*>(sendData);
    auto* destination = static_cast<std::uint8_t*>(recvData);
    std::size_t sent = 0, received = 0;
    try {
        while (sent < sendBytes || received < recvBytes) {
            remainingMs(deadline);
            bool progress = false;
            if (sent < sendBytes) {
                const auto n = ::send(sender.socket_, source + sent,
                    std::min(sendBytes - sent, static_cast<std::size_t>(INT_MAX)),
                    MSG_NOSIGNAL | MSG_DONTWAIT);
                if (n > 0) {
                    sent += static_cast<std::size_t>(n);
                    sender.sent_ += static_cast<std::uint64_t>(n);
                    progress = true;
                } else if (n == 0) {
                    systemFailure("RSS send made no progress", EPIPE);
                } else if (errno == EINTR) {
                    progress = true;
                } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
                    systemFailure("RSS send");
                }
            }
            if (received < recvBytes) {
                const auto n = ::recv(receiver.socket_, destination + received,
                    std::min(recvBytes - received, static_cast<std::size_t>(INT_MAX)), MSG_DONTWAIT);
                if (n > 0) {
                    received += static_cast<std::size_t>(n);
                    receiver.received_ += static_cast<std::uint64_t>(n);
                    progress = true;
                } else if (n == 0) {
                    systemFailure("RSS peer closed before the requested data arrived", ECONNRESET);
                } else if (errno == EINTR) {
                    progress = true;
                } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
                    systemFailure("RSS recv");
                }
            }
            if (progress || (sent == sendBytes && received == recvBytes)) continue;
            std::array<pollfd, 2> events{};
            nfds_t count = 0;
            if (sent < sendBytes) events[count++] = {sender.socket_, POLLOUT, 0};
            if (received < recvBytes) {
                if (count && sender.socket_ == receiver.socket_) events[0].events |= POLLIN;
                else events[count++] = {receiver.socket_, POLLIN, 0};
            }
            pollUntil(events.data(), count, deadline);
        }
    } catch (...) {
        sender.close();
        receiver.close();
        throw;
    }
}

void Peer3PC::sendUntil(const void* data, std::size_t bytes, Deadline deadline)
{
    exchangeUntil(*this, data, bytes, *this, nullptr, 0, deadline);
}

void Peer3PC::recvUntil(void* data, std::size_t bytes, Deadline deadline)
{
    exchangeUntil(*this, nullptr, 0, *this, data, bytes, deadline);
}

void Peer3PC::send_bytes(const void* data, std::size_t bytes)
{
    sendUntil(data, bytes, deadlineAfter(timeout_));
}

void Peer3PC::recv_bytes(void* data, std::size_t bytes)
{
    recvUntil(data, bytes, deadlineAfter(timeout_));
}

void Peer3PC::send_ge(const GroupElement& value, int bw) { send_ge_array(&value, 1, bw); }
GroupElement Peer3PC::recv_ge(int bw)
{
    GroupElement value{};
    recv_ge_array(&value, 1, bw);
    return value;
}

void Peer3PC::send_ge_array(const GroupElement* data, std::size_t count, int bw)
{
    const auto deadline = deadlineAfter(timeout_);
    const auto wire = encodeElements(data, count, bw);
    sendUntil(wire.data(), wire.size(), deadline);
}

void Peer3PC::recv_ge_array(GroupElement* data, std::size_t count, int bw)
{
    const auto deadline = deadlineAfter(timeout_);
    checkedBytes(data, count, sizeof(GroupElement));
    Bytes wire(checkedBytes(data, count, elementBytes(bw)));
    recvUntil(wire.data(), wire.size(), deadline);
    decodeElements(wire, data, count, bw);
}

void Peer3PC::send_block(const osuCrypto::block& value) { send_block_array(&value, 1); }
osuCrypto::block Peer3PC::recv_block()
{
    osuCrypto::block value;
    recv_block_array(&value, 1);
    return value;
}

void Peer3PC::send_block_array(const osuCrypto::block* data, std::size_t count)
{
    static_assert(sizeof(osuCrypto::block) == 16, "RSS requires 128-bit blocks");
    send_bytes(data, checkedBytes(data, count, 16));
}

void Peer3PC::recv_block_array(osuCrypto::block* data, std::size_t count)
{
    recv_bytes(data, checkedBytes(data, count, 16));
}

void Peer3PC::send_rss_share(const RssShare& value, int bw) { send_rss_share_array(&value, 1, bw); }
RssShare Peer3PC::recv_rss_share(int bw)
{
    RssShare value;
    recv_rss_share_array(&value, 1, bw);
    return value;
}

void Peer3PC::send_rss_share_array(const RssShare* data, std::size_t count, int bw)
{
    const auto deadline = deadlineAfter(timeout_);
    const auto wire = encodeShares(data, count, bw);
    sendUntil(wire.data(), wire.size(), deadline);
}

void Peer3PC::recv_rss_share_array(RssShare* data, std::size_t count, int bw)
{
    const auto deadline = deadlineAfter(timeout_);
    checkedBytes(data, count, sizeof(RssShare));
    Bytes wire(checkedBytes(data, count, 2 * elementBytes(bw)));
    recvUntil(wire.data(), wire.size(), deadline);
    decodeShares(wire, data, count, bw);
}

void Peer3PC::send_rss_xor_share(const RssXorShare& value, int bw) { send_rss_xor_share_array(&value, 1, bw); }
RssXorShare Peer3PC::recv_rss_xor_share(int bw)
{
    RssXorShare value;
    recv_rss_xor_share_array(&value, 1, bw);
    return value;
}

void Peer3PC::send_rss_xor_share_array(const RssXorShare* data, std::size_t count, int bw)
{
    const auto deadline = deadlineAfter(timeout_);
    const auto wire = encodeShares(data, count, bw);
    sendUntil(wire.data(), wire.size(), deadline);
}

void Peer3PC::recv_rss_xor_share_array(RssXorShare* data, std::size_t count, int bw)
{
    const auto deadline = deadlineAfter(timeout_);
    checkedBytes(data, count, sizeof(RssXorShare));
    Bytes wire(checkedBytes(data, count, 2 * elementBytes(bw)));
    recvUntil(wire.data(), wire.size(), deadline);
    decodeShares(wire, data, count, bw);
}

void Peer3PC::sync()
{
    const std::uint8_t outgoing = 0xa5;
    std::uint8_t incoming = 0;
    exchangeUntil(*this, &outgoing, 1, *this, &incoming, 1, deadlineAfter(timeout_));
    if (incoming != outgoing) {
        close();
        throw std::runtime_error("RSS pair synchronization mismatch");
    }
}

Comm3PC::Comm3PC(RssParty self, const std::array<Endpoint3PC, 3>& endpoints,
                 Comm3PCOptions options)
    : self_(self), options_(options)
{
    const int own = partyIndex(self);
    const auto deadline = deadlineAfter(options_.connectTimeout);
    deadlineAfter(options_.ioTimeout);
    std::array<sockaddr_in, 3> addresses{};
    for (int p = 0; p < 3; ++p) {
        if (endpoints[p].port == 0) throw std::invalid_argument("RSS TCP port must be nonzero");
        addresses[p].sin_family = AF_INET;
        addresses[p].sin_port = htons(endpoints[p].port);
        if (::inet_pton(AF_INET, endpoints[p].ip.c_str(), &addresses[p].sin_addr) != 1) {
            throw std::invalid_argument("RSS endpoint must contain a numeric IPv4 address");
        }
        if (addresses[p].sin_addr.s_addr == htonl(INADDR_ANY)) {
            throw std::invalid_argument("RSS endpoint must be a connectable IPv4 address");
        }
        for (int previous = 0; previous < p; ++previous) {
            if (addresses[p].sin_addr.s_addr == addresses[previous].sin_addr.s_addr &&
                addresses[p].sin_port == addresses[previous].sin_port) {
                throw std::invalid_argument("RSS endpoints must be distinct");
            }
        }
    }

    std::string phase = "binding listener";
    try {
        OwnedFd listener(tcpSocket());
        const int one = 1;
        if (::setsockopt(listener.get(), SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) < 0) {
            systemFailure("RSS SO_REUSEADDR");
        }
        sockaddr_in local = addresses[own];
        local.sin_addr.s_addr = htonl(INADDR_ANY);
        if (::bind(listener.get(), reinterpret_cast<const sockaddr*>(&local), sizeof(local)) < 0) {
            systemFailure("RSS bind");
        }
        if (::listen(listener.get(), 4) < 0) systemFailure("RSS listen");

        // This party starts its listener before making outgoing connections.
        // Ascending lower IDs prevent connection cycles; accept identifies higher
        // IDs from their handshake, not from the order in which sockets arrive.
        for (int other = 0; other < own; ++other) {
            phase = "connecting to P" + std::to_string(other);
            for (;;) {
                std::unique_ptr<Peer3PC> remote;
                try {
                    OwnedFd connected(connectUntil(addresses[other], addresses[own], deadline));
                    // new allocates before release(); the constructor owns fd on failure.
                    remote.reset(new Peer3PC(connected.release(), options_.ioTimeout));
                    const auto attempt = std::min(deadline, deadlineAfter(
                        std::min(options_.ioTimeout, std::chrono::milliseconds{1000})));
                    const auto request = hello(own, other);
                    std::array<std::uint8_t, 8> reply{};
                    remote->sendUntil(request.data(), request.size(), attempt);
                    remote->recvUntil(reply.data(), reply.size(), attempt);
                    if (checkHello(reply, own) != other) {
                        throw std::runtime_error("RSS connected to wrong party");
                    }
                    const std::uint8_t confirmed = 0xc3;
                    remote->sendUntil(&confirmed, 1, attempt);
                    peers_[other] = std::move(remote);
                    break;
                } catch (const std::system_error& error) {
                    // Refusal/reset can surface after connect(), before the peer's
                    // listener is ready (including on WSL localhost forwarding).
                    if (!retryableConnect(error.code().value()) && error.code().value() != EPIPE) throw;
                    remote.reset();
                    retryPause(deadline);
                }
            }
        }
        int accepted = own + 1;
        while (accepted < 3) {
            phase = "accepting higher-numbered parties";
            std::unique_ptr<Peer3PC> remote;
            try {
                OwnedFd connected(acceptUntil(listener.get(), deadline));
                remote.reset(new Peer3PC(connected.release(), options_.ioTimeout));
                phase = "receiving incoming handshake";
                const auto attempt = std::min(deadline, deadlineAfter(
                    std::min(options_.ioTimeout, std::chrono::milliseconds{1000})));
                std::array<std::uint8_t, 8> request{};
                remote->recvUntil(request.data(), request.size(), attempt);
                const int other = checkHello(request, own);
                if (other <= own || peers_[other]) {
                    throw std::runtime_error("RSS duplicate or unexpected party");
                }
                const auto reply = hello(own, other);
                remote->sendUntil(reply.data(), reply.size(), attempt);
                std::uint8_t confirmation = 0;
                remote->recvUntil(&confirmation, 1, attempt);
                if (confirmation != 0xc3) throw std::runtime_error("RSS handshake confirmation mismatch");
                peers_[other] = std::move(remote);
                ++accepted;
            } catch (const std::system_error& error) {
                const int reason = error.code().value();
                if (reason != ECONNRESET && reason != EPIPE &&
                    !(reason == ETIMEDOUT && Clock::now() < deadline)) throw;
                remote.reset();
            }
        }

        // Wait until every party has both edges before exchanging seeds.
        phase = "waiting for complete mesh";
        barrierUntil(deadline);
        phase = "exchanging shared seeds";
        for (int other = 0; other < 3; ++other) {
            if (other == own) continue;
            std::array<std::uint8_t, 32> seeds{};
            if (own < other) {
                systemRandom(seeds.data(), seeds.size());
                peers_[other]->sendUntil(seeds.data(), seeds.size(), deadline);
            } else {
                peers_[other]->recvUntil(seeds.data(), seeds.size(), deadline);
            }
            osuCrypto::block arithmeticSeed, booleanSeed;
            std::memcpy(&arithmeticSeed, seeds.data(), 16);
            std::memcpy(&booleanSeed, seeds.data() + 16, 16);
            if (other == (own + 2) % 3) {
                prevArithmetic_.SetSeed(arithmeticSeed);
                prevBoolean_.SetSeed(booleanSeed);
            } else {
                nextArithmetic_.SetSeed(arithmeticSeed);
                nextBoolean_.SetSeed(booleanSeed);
            }
        }
        phase = "waiting for PRNG initialization";
        barrierUntil(deadline);
        ready_ = true;
    } catch (const std::system_error& error) {
        throw std::system_error(error.code(), "RSS P" + std::to_string(own) + " " + phase);
    }

}

Comm3PC::~Comm3PC() { close(); }

RssParty Comm3PC::party() const noexcept { return self_; }
RssParty Comm3PC::prevParty() const noexcept
{
    return static_cast<RssParty>((static_cast<int>(self_) + 2) % 3);
}
RssParty Comm3PC::nextParty() const noexcept
{
    return static_cast<RssParty>((static_cast<int>(self_) + 1) % 3);
}

void Comm3PC::requireOpen() const
{
    if (!ready_) throw std::logic_error("RSS communication context is closed");
    for (const auto& connection : peers_) if (connection) connection->requireOpen();
}

Peer3PC& Comm3PC::peer(RssParty other)
{
    const int index = partyIndex(other);
    if (other == self_) throw std::invalid_argument("RSS cannot communicate with itself");
    requireOpen();
    return *peers_[index];
}

Peer3PC& Comm3PC::prev() { return peer(prevParty()); }
Peer3PC& Comm3PC::next() { return peer(nextParty()); }

void Comm3PC::exchange_bytes(RssParty sendTo, const void* sendData, std::size_t sendBytes,
                            RssParty recvFrom, void* recvData, std::size_t recvBytes)
{
    // Complete all validation before either direction can make progress.
    checkedBytes(sendData, sendBytes, 1);
    checkedBytes(recvData, recvBytes, 1);
    checkRawOverlap(sendData, sendBytes, recvData, recvBytes);
    auto& sender = peer(sendTo);
    auto& receiver = peer(recvFrom);
    try {
        Peer3PC::exchangeUntil(sender, sendData, sendBytes, receiver, recvData, recvBytes,
                              deadlineAfter(options_.ioTimeout));
    } catch (...) {
        close();
        throw;
    }
}

void Comm3PC::exchange_ge_array(RssParty sendTo, const GroupElement* sendData, std::size_t sendCount,
                               RssParty recvFrom, GroupElement* recvData, std::size_t recvCount,
                               int bw)
{
    const auto deadline = deadlineAfter(options_.ioTimeout);
    const auto width = elementBytes(bw);
    checkedBytes(sendData, sendCount, sizeof(GroupElement));
    checkedBytes(recvData, recvCount, sizeof(GroupElement));
    const auto recvSize = checkedBytes(recvData, recvCount, width);
    checkedBytes(sendData, sendCount, width);
    auto& sender = peer(sendTo);
    auto& receiver = peer(recvFrom);
    const auto outgoing = encodeElements(sendData, sendCount, bw);
    Bytes incoming(recvSize);
    try {
        Peer3PC::exchangeUntil(sender, outgoing.data(), outgoing.size(),
                              receiver, incoming.data(), incoming.size(), deadline);
    } catch (...) {
        close();
        throw;
    }
    decodeElements(incoming, recvData, recvCount, bw);
}

void Comm3PC::barrierUntil(Deadline deadline)
{
    constexpr std::uint8_t arrived = 0xa1, released = 0xb0;
    std::uint8_t token = 0;
    if (self_ == RssParty::P0) {
        for (int other = 1; other < 3; ++other) {
            peers_[other]->recvUntil(&token, 1, deadline);
            if (token != arrived) throw std::runtime_error("RSS barrier arrival mismatch");
        }
        for (int other = 1; other < 3; ++other) peers_[other]->sendUntil(&released, 1, deadline);
    } else {
        peers_[0]->sendUntil(&arrived, 1, deadline);
        peers_[0]->recvUntil(&token, 1, deadline);
        if (token != released) throw std::runtime_error("RSS barrier release mismatch");
    }
}

void Comm3PC::sync()
{
    requireOpen();
    try {
        barrierUntil(deadlineAfter(options_.ioTimeout));
    } catch (...) {
        close();
        throw;
    }
}

void Comm3PC::close() noexcept
{
    ready_ = false;
    for (auto& connection : peers_) if (connection) connection->close();
}

std::uint64_t Comm3PC::bytesSent() const noexcept
{
    std::uint64_t bytes = 0;
    for (const auto& connection : peers_) if (connection) bytes += connection->bytesSent();
    return bytes;
}

std::uint64_t Comm3PC::bytesReceived() const noexcept
{
    std::uint64_t bytes = 0;
    for (const auto& connection : peers_) if (connection) bytes += connection->bytesReceived();
    return bytes;
}

void Comm3PC::zeroBytesSent() noexcept
{
    for (auto& connection : peers_) if (connection) connection->zeroBytesSent();
}

void Comm3PC::zeroBytesReceived() noexcept
{
    for (auto& connection : peers_) if (connection) connection->zeroBytesReceived();
}

osuCrypto::PRNG& Comm3PC::prevArithmeticPrng() { requireOpen(); return prevArithmetic_; }
osuCrypto::PRNG& Comm3PC::nextArithmeticPrng() { requireOpen(); return nextArithmetic_; }
osuCrypto::PRNG& Comm3PC::prevBooleanPrng() { requireOpen(); return prevBoolean_; }
osuCrypto::PRNG& Comm3PC::nextBooleanPrng() { requireOpen(); return nextBoolean_; }
