#pragma once

#include <crypto/primitives/RSS/group_element.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct Endpoint3PC {
    // Keep same-WSL traffic on Linux loopback, away from host localhost forwarding.
    // Cross-machine callers supply each machine's explicit IPv4 address.
    std::string ip = "127.0.0.2";
    std::uint16_t port = 0;
};

struct Comm3PCOptions {
    std::chrono::milliseconds connectTimeout{30000};
    std::chrono::milliseconds ioTimeout{30000};
};

// A single full-duplex connection. No message framing: callers must agree on
// type, count, bit width and ordering. Concurrent calls are not supported.
class Peer3PC {
public:
    // Takes ownership of an already connected socket, including on failure.
    // Comm3PC normally constructs peers; this entry also permits socketpair tests.
    explicit Peer3PC(int connectedSocket,
                     std::chrono::milliseconds ioTimeout = std::chrono::milliseconds{30000});
    ~Peer3PC();
    Peer3PC(const Peer3PC&) = delete;
    Peer3PC& operator=(const Peer3PC&) = delete;
    Peer3PC(Peer3PC&&) = delete;
    Peer3PC& operator=(Peer3PC&&) = delete;

    void send_bytes(const void* data, std::size_t bytes);
    void recv_bytes(void* data, std::size_t bytes);
    void send_ge(const GroupElement& value, int bw);
    GroupElement recv_ge(int bw);
    void send_ge_array(const GroupElement* data, std::size_t count, int bw);
    void recv_ge_array(GroupElement* data, std::size_t count, int bw);
    void send_block(const osuCrypto::block& value);
    osuCrypto::block recv_block();
    void send_block_array(const osuCrypto::block* data, std::size_t count);
    void recv_block_array(osuCrypto::block* data, std::size_t count);
    void send_rss_share(const RssShare& value, int bw);
    RssShare recv_rss_share(int bw);
    void send_rss_share_array(const RssShare* data, std::size_t count, int bw);
    void recv_rss_share_array(RssShare* data, std::size_t count, int bw);
    void send_rss_xor_share(const RssXorShare& value, int bw);
    RssXorShare recv_rss_xor_share(int bw);
    void send_rss_xor_share_array(const RssXorShare* data, std::size_t count, int bw);
    void recv_rss_xor_share_array(RssXorShare* data, std::size_t count, int bw);

    // Both endpoints must call sync on the same edge.
    void sync();
    void close() noexcept;
    std::uint64_t bytesSent() const noexcept;
    std::uint64_t bytesReceived() const noexcept;
    void zeroBytesSent() noexcept;
    void zeroBytesReceived() noexcept;

private:
    friend class Comm3PC;
    using Deadline = std::chrono::steady_clock::time_point;
    int socket_ = -1;
    std::chrono::milliseconds timeout_;
    std::uint64_t sent_ = 0;
    std::uint64_t received_ = 0;

    void requireOpen() const;
    void sendUntil(const void* data, std::size_t bytes, Deadline deadline);
    void recvUntil(void* data, std::size_t bytes, Deadline deadline);
    static void exchangeUntil(Peer3PC& sender, const void* sendData, std::size_t sendBytes,
                              Peer3PC& receiver, void* recvData, std::size_t recvBytes,
                              Deadline deadline);
};

// Three equal RSS participants. Construction completes the mesh and initializes
// independent arithmetic / boolean edge streams. TCP is plaintext; handshakes
// detect configuration mistakes, not malicious participants or network attackers.
class Comm3PC {
public:
    Comm3PC(RssParty self, const std::array<Endpoint3PC, 3>& endpoints,
            Comm3PCOptions options = {});
    ~Comm3PC();
    Comm3PC(const Comm3PC&) = delete;
    Comm3PC& operator=(const Comm3PC&) = delete;
    Comm3PC(Comm3PC&&) = delete;
    Comm3PC& operator=(Comm3PC&&) = delete;

    RssParty party() const noexcept;
    RssParty prevParty() const noexcept;
    RssParty nextParty() const noexcept;
    Peer3PC& peer(RssParty other);
    Peer3PC& prev();
    Peer3PC& next();

    // Progress both directions together, even when the peers differ.
    // Raw buffers must not overlap. The typed variant supports in-place exchange.
    void exchange_bytes(RssParty sendTo, const void* sendData, std::size_t sendBytes,
                        RssParty recvFrom, void* recvData, std::size_t recvBytes);
    void exchange_ge_array(RssParty sendTo, const GroupElement* sendData, std::size_t sendCount,
                           RssParty recvFrom, GroupElement* recvData, std::size_t recvCount,
                           int bw);

    // All parties must call this barrier in the same order.
    void sync();
    void close() noexcept;
    std::uint64_t bytesSent() const noexcept;
    std::uint64_t bytesReceived() const noexcept;
    void zeroBytesSent() noexcept;
    void zeroBytesReceived() noexcept;

    // Consume matching streams in matching order at both edge endpoints.
    // prev generates x_i; next generates x_(i+1). Input-owner randomness is separate.
    osuCrypto::PRNG& prevArithmeticPrng();
    osuCrypto::PRNG& nextArithmeticPrng();
    osuCrypto::PRNG& prevBooleanPrng();
    osuCrypto::PRNG& nextBooleanPrng();

private:
    using Deadline = std::chrono::steady_clock::time_point;
    RssParty self_;
    Comm3PCOptions options_;
    std::array<std::unique_ptr<Peer3PC>, 3> peers_;
    osuCrypto::PRNG prevArithmetic_, nextArithmetic_, prevBoolean_, nextBoolean_;
    bool ready_ = false;

    void requireOpen() const;
    void barrierUntil(Deadline deadline);
};
