#include <crypto/commom/comms-3PC.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <dirent.h>
#include <functional>
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace std::chrono_literals;

namespace {
using Job = std::function<void()>;
std::string stage;
volatile sig_atomic_t signalsSeen = 0;

static_assert(!std::is_copy_constructible<Peer3PC>::value &&
              !std::is_move_constructible<Peer3PC>::value, "Peers must have stable ownership");
static_assert(!std::is_copy_constructible<Comm3PC>::value &&
              !std::is_move_constructible<Comm3PC>::value, "Contexts must have stable ownership");

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(stage + ": " + message);
}

template<class F> void expectInvalid(F operation)
{
    bool caught = false;
    try { operation(); } catch (const std::invalid_argument&) { caught = true; }
    require(caught, "invalid argument was not rejected");
}

template<class F> void expectFailure(F operation)
{
    bool caught = false;
    try { operation(); } catch (const std::exception&) { caught = true; }
    require(caught, "expected communication failure");
}

GroupElement maskFor(int bw)
{
    return std::numeric_limits<GroupElement>::max() >> (64 - bw);
}

std::size_t widthFor(int bw) { return bw <= 8 ? 1 : bw <= 16 ? 2 : bw <= 32 ? 4 : 8; }
RssParty party(int p) { return static_cast<RssParty>(p); }

void closeFds(std::vector<int>& fds)
{
    for (int fd : fds) if (fd >= 0) ::close(fd);
    fds.clear();
}

void rawRead(int fd, void* buffer, std::size_t bytes)
{
    auto* out = static_cast<unsigned char*>(buffer);
    while (bytes) {
        const auto n = ::read(fd, out, bytes);
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, "raw read failed");
        out += n;
        bytes -= static_cast<std::size_t>(n);
    }
}

void rawWrite(int fd, const void* buffer, std::size_t bytes)
{
    const auto* in = static_cast<const unsigned char*>(buffer);
    while (bytes) {
        const auto n = ::write(fd, in, bytes);
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, "raw write failed");
        in += n;
        bytes -= static_cast<std::size_t>(n);
    }
}

void rawSend(int fd, const void* buffer, std::size_t bytes)
{
    const auto* in = static_cast<const unsigned char*>(buffer);
    while (bytes) {
        const auto n = ::send(fd, in, bytes, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, "raw send failed");
        in += n;
        bytes -= static_cast<std::size_t>(n);
    }
}

// Reserved ports stay bound until every child has closed its inherited copies.
// The parent releases them immediately before letting the children run.
struct Ports {
    std::array<Endpoint3PC, 3> endpoints{};
    std::vector<int> fds;
    Ports()
    {
        try {
            for (int p = 0; p < 3; ++p) {
                const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
                require(fd >= 0, "port reservation socket");
                fds.push_back(fd);
                sockaddr_in address{};
                address.sin_family = AF_INET;
                require(::inet_pton(AF_INET, endpoints[p].ip.c_str(), &address.sin_addr) == 1,
                        "test loopback address");
                require(::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0,
                        "port reservation bind");
                socklen_t size = sizeof(address);
                require(::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &size) == 0,
                        "port reservation getsockname");
                endpoints[p].port = ntohs(address.sin_port);
            }
        } catch (...) { closeFds(fds); throw; }
    }
    ~Ports() { closeFds(fds); }
    std::vector<int> release() { auto result = std::move(fds); fds.clear(); return result; }
};

void runJobs(const std::vector<Job>& jobs, std::vector<int> inherited = {},
             bool closeInChildren = false)
{
    int ready[2]{-1, -1}, start[2]{-1, -1};
    std::vector<pid_t> children;
    auto cleanup = [&] {
        for (pid_t child : children) if (child > 0) ::kill(child, SIGKILL);
        for (pid_t child : children) {
            if (child > 0) while (::waitpid(child, nullptr, 0) < 0 && errno == EINTR) {}
        }
        for (int fd : {ready[0], ready[1], start[0], start[1]}) if (fd >= 0) ::close(fd);
        closeFds(inherited);
    };
    try {
        require(::pipe(ready) == 0 && ::pipe(start) == 0, "child coordination pipes");
        for (std::size_t i = 0; i < jobs.size(); ++i) {
            const pid_t child = ::fork();
            require(child >= 0, "fork");
            if (child == 0) {
                ::close(ready[0]);
                ::close(start[1]);
                if (closeInChildren) closeFds(inherited);
                try {
                    const char marker = 'r';
                    rawWrite(ready[1], &marker, 1);
                    ::close(ready[1]);
                    char go;
                    rawRead(start[0], &go, 1);
                    ::close(start[0]);
                    jobs[i]();
                    ::_exit(0);
                } catch (const std::exception& error) {
                    std::cerr << "RSS comms child " << i << " [" << stage << "]: " << error.what() << '\n';
                    ::_exit(1);
                }
            }
            children.push_back(child);
        }
        ::close(ready[1]); ready[1] = -1;
        ::close(start[0]); start[0] = -1;
        for (std::size_t i = 0; i < jobs.size(); ++i) {
            pollfd event{ready[0], POLLIN, 0};
            require(::poll(&event, 1, 3000) > 0, "child readiness timeout");
            char marker;
            rawRead(ready[0], &marker, 1);
        }
        ::close(ready[0]); ready[0] = -1;
        closeFds(inherited);
        for (std::size_t i = 0; i < jobs.size(); ++i) {
            const char go = 'g';
            rawWrite(start[1], &go, 1);
        }
        ::close(start[1]); start[1] = -1;
        const auto deadline = std::chrono::steady_clock::now() + 45s;
        std::size_t remaining = children.size();
        while (remaining) {
            require(std::chrono::steady_clock::now() < deadline, "child process timeout");
            for (auto& child : children) {
                if (child <= 0) continue;
                int status = 0;
                const auto result = ::waitpid(child, &status, WNOHANG);
                if (result < 0 && errno == EINTR) continue;
                require(result >= 0, "waitpid");
                if (result > 0) {
                    child = -1;
                    --remaining;
                    require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "child failed");
                }
            }
            if (remaining) std::this_thread::sleep_for(2ms);
        }
    } catch (...) { cleanup(); throw; }
    cleanup();
}

std::array<GroupElement, 7> values(int sender, int bw)
{
    return {{0, 1, maskFor(bw), maskFor(bw) + 1,
             GroupElement{1} << (bw - 1), ~GroupElement{0},
             0xfedcba9876543210ULL ^ static_cast<GroupElement>(sender)}};
}

void sendPacket(Peer3PC& peer, int sender, int bw)
{
    const auto v = values(sender, bw);
    const RssShare a{v[6], v[5]};
    const std::array<RssShare, 3> aa{{{v[0], v[1]}, {v[2], v[3]}, {v[4], v[6]}}};
    const RssXorShare b{v[5], v[6]};
    const std::array<RssXorShare, 3> bb{{{v[1], v[0]}, {v[3], v[2]}, {v[6], v[4]}}};
    const auto block = osuCrypto::toBlock(0x123456789abcdef0ULL, sender + 17);
    const std::array<osuCrypto::block, 2> blocks{{block, osuCrypto::toBlock(sender + 33, 99)}};
    std::array<unsigned char, 17> raw{};
    for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = static_cast<unsigned char>(i + sender);
    peer.send_ge(v[6], bw);
    peer.send_ge_array(v.data(), v.size(), bw);
    peer.send_rss_share(a, bw);
    peer.send_rss_share_array(aa.data(), aa.size(), bw);
    peer.send_rss_xor_share(b, bw);
    peer.send_rss_xor_share_array(bb.data(), bb.size(), bw);
    peer.send_block(block);
    peer.send_block_array(blocks.data(), blocks.size());
    peer.send_bytes(raw.data(), raw.size());
}

template<class Share>
void checkShare(Share value, GroupElement first, GroupElement second, int bw)
{
    require(value.first == (first & maskFor(bw)), "wrong first share field");
    require(value.second == (second & maskFor(bw)), "wrong second share field");
}

void receivePacket(Peer3PC& peer, int sender, int bw)
{
    const auto v = values(sender, bw);
    require(peer.recv_ge(bw) == (v[6] & maskFor(bw)), "scalar ring value mismatch");
    std::array<GroupElement, 7> got{};
    peer.recv_ge_array(got.data(), got.size(), bw);
    for (std::size_t i = 0; i < got.size(); ++i) {
        require(got[i] == (v[i] & maskFor(bw)), "ring array value mismatch");
    }
    checkShare(peer.recv_rss_share(bw), v[6], v[5], bw);
    std::array<RssShare, 3> a{};
    peer.recv_rss_share_array(a.data(), a.size(), bw);
    checkShare(a[0], v[0], v[1], bw); checkShare(a[1], v[2], v[3], bw);
    checkShare(a[2], v[4], v[6], bw);
    checkShare(peer.recv_rss_xor_share(bw), v[5], v[6], bw);
    std::array<RssXorShare, 3> b{};
    peer.recv_rss_xor_share_array(b.data(), b.size(), bw);
    checkShare(b[0], v[1], v[0], bw); checkShare(b[1], v[3], v[2], bw);
    checkShare(b[2], v[6], v[4], bw);
    const auto expected = osuCrypto::toBlock(0x123456789abcdef0ULL, sender + 17);
    const auto block = peer.recv_block();
    require(std::memcmp(&block, &expected, 16) == 0, "block mismatch");
    std::array<osuCrypto::block, 2> blocks{};
    peer.recv_block_array(blocks.data(), blocks.size());
    const auto second = osuCrypto::toBlock(sender + 33, 99);
    require(std::memcmp(&blocks[0], &expected, 16) == 0 &&
            std::memcmp(&blocks[1], &second, 16) == 0, "block array mismatch");
    std::array<unsigned char, 17> raw{};
    peer.recv_bytes(raw.data(), raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        require(raw[i] == static_cast<unsigned char>(i + sender), "raw bytes mismatch");
    }
}

void emptyArrays(Peer3PC& peer)
{
    const auto sent = peer.bytesSent(), received = peer.bytesReceived();
    peer.send_bytes(nullptr, 0); peer.recv_bytes(nullptr, 0);
    peer.send_ge_array(nullptr, 0, 1); peer.recv_ge_array(nullptr, 0, 64);
    peer.send_block_array(nullptr, 0); peer.recv_block_array(nullptr, 0);
    peer.send_rss_share_array(nullptr, 0, 7); peer.recv_rss_share_array(nullptr, 0, 31);
    peer.send_rss_xor_share_array(nullptr, 0, 9); peer.recv_rss_xor_share_array(nullptr, 0, 33);
    require(sent == peer.bytesSent() && received == peer.bytesReceived(), "empty array transmitted data");
}

GroupElement bigValue(int sender, std::size_t index)
{
    return (GroupElement{0x9876543210abcdefULL} * (index + 1)) ^ (GroupElement(sender + 1) << 47);
}

void meshNode(int own, const std::array<Endpoint3PC, 3>& endpoints, int delay, bool full)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    stage = "mesh party " + std::to_string(own);
    Comm3PC comm(party(own), endpoints);
    require(comm.party() == party(own) && comm.prevParty() == party((own + 2) % 3)
            && comm.nextParty() == party((own + 1) % 3), "neighbor mapping");
    require(comm.bytesSent() > 0 && comm.bytesReceived() > 0, "setup bytes not counted");
    expectInvalid([&] { comm.peer(party(own)); });
    expectInvalid([&] { comm.peer(party(-1)); });
    expectInvalid([&] { comm.peer(party(3)); });
    GroupElement scratch = 1;
    const auto initialSent = comm.bytesSent(), initialReceived = comm.bytesReceived();
    expectInvalid([&] { comm.exchange_bytes(comm.nextParty(), &scratch, 1, comm.prevParty(), &scratch, 1); });
    expectInvalid([&] { comm.exchange_ge_array(comm.nextParty(), &scratch, 1, party(own), &scratch, 1, 64); });
    expectInvalid([&] { comm.exchange_ge_array(comm.nextParty(), &scratch, 1, comm.prevParty(), nullptr, 1, 64); });
    expectInvalid([&] { comm.exchange_ge_array(comm.nextParty(), &scratch, 1, comm.prevParty(), &scratch, 1, 0); });
    require(initialSent == comm.bytesSent() && initialReceived == comm.bytesReceived(),
            "invalid exchange performed I/O");
    comm.exchange_bytes(comm.nextParty(), nullptr, 0, comm.prevParty(), nullptr, 0);
    comm.exchange_ge_array(comm.nextParty(), nullptr, 0, comm.prevParty(), nullptr, 0, 64);
    comm.sync();

    for (int low = 0; low < 3; ++low) {
        for (int high = low + 1; high < 3; ++high) {
            if (own != low && own != high) continue;
            auto& edge = comm.peer(party(own == low ? high : low));
            edge.zeroBytesSent(); edge.zeroBytesReceived();
            emptyArrays(edge);
            for (int bw = 1; bw <= 64; ++bw) {
                stage = "mesh scalar/array bw=" + std::to_string(bw);
                const auto sent = edge.bytesSent(), received = edge.bytesReceived();
                if (own == low) { sendPacket(edge, own, bw); receivePacket(edge, high, bw); }
                else { receivePacket(edge, low, bw); sendPacket(edge, own, bw); }
                const auto expectedBytes = 24 * widthFor(bw) + 65;
                require(edge.bytesSent() - sent == expectedBytes &&
                        edge.bytesReceived() - received == expectedBytes, "payload byte counters");
            }
            const auto sent = edge.bytesSent(), received = edge.bytesReceived();
            edge.sync();
            require(edge.bytesSent() == sent + 1 && edge.bytesReceived() == received + 1,
                    "pair synchronization byte counters");
        }
    }
    comm.sync();

    stage = "pairwise PRNG initialization and RSS samples";
    // Local reference PRNGs prove setup did not consume any business output.
    osuCrypto::PRNG refPA(comm.prevArithmeticPrng().getSeed());
    osuCrypto::PRNG refNA(comm.nextArithmeticPrng().getSeed());
    osuCrypto::PRNG refPB(comm.prevBooleanPrng().getSeed());
    osuCrypto::PRNG refNB(comm.nextBooleanPrng().getSeed());
    bool distinctDomains = false;
    for (int bw = 1; bw <= 64; ++bw) {
        for (int n = 0; n < 12; ++n) {
            const auto a = randomShareRss(bw, comm.prevArithmeticPrng(), comm.nextArithmeticPrng());
            const auto b = randomShareRssXor(bw, comm.prevBooleanPrng(), comm.nextBooleanPrng());
            require(a.first == (refPA.get<GroupElement>() & maskFor(bw)) &&
                    a.second == (refNA.get<GroupElement>() & maskFor(bw)), "arithmetic PRNG consumed or misrouted");
            require(b.first == (refPB.get<GroupElement>() & maskFor(bw)) &&
                    b.second == (refNB.get<GroupElement>() & maskFor(bw)), "boolean PRNG consumed or misrouted");
            distinctDomains |= (a.first != b.first || a.second != b.second);
            const GroupElement send[2]{a.first, b.first};
            GroupElement incoming[2]{};
            comm.exchange_ge_array(comm.prevParty(), send, 2, comm.nextParty(), incoming, 2, bw);
            require(a.second == incoming[0] && b.second == incoming[1], "neighbor PRNG streams differ");
        }
    }
    require(distinctDomains, "arithmetic and boolean streams were reused");

    if (full) {
        stage = "large ring exchange";
        constexpr std::size_t count = 1 << 20; // 8 MiB, larger than socket buffers.
        std::vector<GroupElement> data(count), incoming(count);
        for (std::size_t i = 0; i < count; ++i) data[i] = bigValue(own, i);
        comm.sync(); comm.zeroBytesSent(); comm.zeroBytesReceived();
        comm.exchange_ge_array(comm.nextParty(), data.data(), count,
                               comm.prevParty(), incoming.data(), count, 37);
        for (std::size_t i = 0; i < count; ++i) {
            require(incoming[i] == (bigValue((own + 2) % 3, i) & maskFor(37)), "ring payload mismatch");
        }
        require(comm.bytesSent() == count * 8 && comm.bytesReceived() == count * 8,
                "large ring byte counters");
        comm.sync();
        stage = "reverse in-place ring exchange";
        comm.exchange_ge_array(comm.prevParty(), data.data(), count,
                               comm.nextParty(), data.data(), count, 64);
        for (std::size_t i = 0; i < count; ++i) {
            require(data[i] == bigValue((own + 1) % 3, i), "in-place ring mismatch");
        }
        for (int low = 0; low < 3; ++low) {
            for (int high = low + 1; high < 3; ++high) {
                if (own != low && own != high) continue;
                stage = "large same-peer duplex exchange";
                const auto other = party(own == low ? high : low);
                std::vector<unsigned char> send(4 << 20, static_cast<unsigned char>(own + 1));
                std::vector<unsigned char> recv(send.size());
                comm.exchange_bytes(other, send.data(), send.size(), other, recv.data(), recv.size());
                require(std::all_of(recv.begin(), recv.end(), [&](unsigned char c) {
                    return c == static_cast<unsigned char>(static_cast<int>(other) + 1);
                }), "duplex payload mismatch");
            }
        }
    }

    stage = "three-party barrier";
    comm.sync();
    if (own == 2) std::this_thread::sleep_for(100ms);
    const auto arrival = static_cast<GroupElement>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    comm.sync();
    const auto returned = static_cast<GroupElement>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    if (own == 2) comm.peer(RssParty::P0).send_ge(arrival, 64);
    if (own == 0) require(returned >= comm.peer(RssParty::P2).recv_ge(64),
                         "barrier returned before P2 arrived");
    comm.sync();
    comm.zeroBytesSent(); comm.zeroBytesReceived();
    require(comm.bytesSent() == 0 && comm.bytesReceived() == 0 &&
            comm.prev().bytesSent() == 0 && comm.next().bytesReceived() == 0, "counter reset");
    comm.close(); comm.close();
    expectFailure([&] { comm.sync(); });
    expectFailure([&] { comm.prevArithmeticPrng(); });
}


void socketPairTest(const std::function<void(int)>& left, const std::function<void(int)>& right)
{
    int sockets[2];
    require(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "socketpair");
    const int small = 4096;
    for (int fd : sockets) {
        require(::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &small, sizeof(small)) == 0, "small send buffer");
    }
    runJobs({
        [=] { ::close(sockets[1]); left(sockets[0]); },
        [=] { ::close(sockets[0]); right(sockets[1]); }
    }, {sockets[0], sockets[1]});
}

void testInvalidAndWire()
{
    stage = "invalid parameters and byte fixtures";
    socketPairTest([](int fd) {
        Peer3PC peer(fd, 2s);
        GroupElement value = 7;
        RssShare a{7, 11};
        RssXorShare b{7, 11};
        const auto max = std::numeric_limits<std::size_t>::max();
        for (int bw : {-1, 0, 65, std::numeric_limits<int>::max()}) {
            expectInvalid([&] { peer.send_ge(value, bw); });
            expectInvalid([&] { peer.recv_ge(bw); });
            expectInvalid([&] { peer.send_ge_array(nullptr, 0, bw); });
            expectInvalid([&] { peer.recv_ge_array(nullptr, 0, bw); });
            expectInvalid([&] { peer.send_rss_share(a, bw); });
            expectInvalid([&] { peer.recv_rss_share(bw); });
            expectInvalid([&] { peer.send_rss_xor_share(b, bw); });
            expectInvalid([&] { peer.recv_rss_xor_share(bw); });
        }
        expectInvalid([&] { peer.send_bytes(nullptr, 1); });
        expectInvalid([&] { peer.recv_bytes(nullptr, 1); });
        expectInvalid([&] { peer.send_ge_array(nullptr, 1, 8); });
        expectInvalid([&] { peer.recv_ge_array(nullptr, 1, 8); });
        expectInvalid([&] { peer.send_rss_share_array(nullptr, 1, 64); });
        expectInvalid([&] { peer.recv_rss_share_array(nullptr, 1, 64); });
        expectInvalid([&] { peer.send_rss_xor_share_array(nullptr, 1, 64); });
        expectInvalid([&] { peer.recv_rss_xor_share_array(nullptr, 1, 64); });
        expectInvalid([&] { peer.send_block_array(nullptr, 1); });
        expectInvalid([&] { peer.recv_block_array(nullptr, 1); });
        expectInvalid([&] { peer.send_ge_array(&value, max, 1); });
        expectInvalid([&] { peer.recv_ge_array(&value, max, 1); });
        expectInvalid([&] { peer.send_rss_share_array(&a, max, 1); });
        expectInvalid([&] { peer.recv_rss_share_array(&a, max, 1); });
        expectInvalid([&] { peer.send_rss_xor_share_array(&b, max, 64); });
        expectInvalid([&] { peer.recv_rss_xor_share_array(&b, max, 64); });
        const auto block = osuCrypto::toBlock(0, 1);
        expectInvalid([&] { peer.send_block_array(&block, max); });
        auto outputBlock = block;
        expectInvalid([&] { peer.recv_block_array(&outputBlock, max); });
        require(peer.bytesSent() == 0 && peer.bytesReceived() == 0, "invalid arguments consumed stream");
        emptyArrays(peer);
        peer.send_ge(0xffff, 9);
        peer.send_rss_share({0x1234, 0xabcd}, 16);
        peer.send_rss_xor_share({0xabcd, 0x1234}, 16);
        peer.send_ge(0x0123456789abcdefULL, 64);
        require(peer.recv_ge(9) == 0x1ff, "receive normalization");
        checkShare(peer.recv_rss_share(9), 0x1ff, 0x100, 9);
        checkShare(peer.recv_rss_xor_share(9), 0x100, 0x1ff, 9);
        peer.sync();
        peer.close(); peer.close();
        require(::fcntl(fd, F_GETFD) == -1 && errno == EBADF, "peer did not close fd");
    }, [](int fd) {
        const std::array<unsigned char, 18> expected{{
            0xff, 0x01, 0x34, 0x12, 0xcd, 0xab, 0xcd, 0xab, 0x34, 0x12,
            0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01
        }};
        std::array<unsigned char, 18> bytes{};
        rawRead(fd, bytes.data(), bytes.size());
        require(bytes == expected, "little-endian wire fixture or field order");
        const std::array<unsigned char, 10> dirty{{0xff, 0xff, 0xff, 0xff, 0x00, 0xff,
                                                 0x00, 0xff, 0xff, 0xff}};
        rawSend(fd, dirty.data(), dirty.size());
        unsigned char sync;
        rawRead(fd, &sync, 1);
        require(sync == 0xa5, "sync fixture");
        rawSend(fd, &sync, 1);
        ::close(fd);
    });
}

void signalHandler(int) { ++signalsSeen; }

void testTransportFailures()
{
    stage = "fragmented large read/write";
    socketPairTest([](int fd) {
        Peer3PC peer(fd, 5s);
        std::vector<unsigned char> data(1 << 20), received(data.size());
        for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<unsigned char>(i * 17);
        peer.send_bytes(data.data(), data.size());
        peer.recv_bytes(received.data(), received.size());
        require(data == received, "fragmented round trip");
        require(peer.bytesSent() == data.size() && peer.bytesReceived() == data.size(), "fragmented counters");
    }, [](int fd) {
        std::vector<unsigned char> data(1 << 20);
        for (std::size_t i = 0; i < data.size();) {
            const auto n = std::min<std::size_t>(127, data.size() - i);
            rawRead(fd, data.data() + i, n);
            i += n;
        }
        for (std::size_t i = 0; i < data.size();) {
            const auto n = std::min<std::size_t>(73, data.size() - i);
            rawSend(fd, data.data() + i, n);
            i += n;
        }
        ::close(fd);
    });

    stage = "EINTR during poll";
    socketPairTest([](int fd) {
        Peer3PC peer(fd, 2s);
        struct sigaction action{};
        action.sa_handler = signalHandler;
        sigemptyset(&action.sa_mask);
        require(::sigaction(SIGALRM, &action, nullptr) == 0, "sigaction");
        signalsSeen = 0;
        itimerval timer{};
        timer.it_value.tv_usec = 10000;
        timer.it_interval.tv_usec = 10000;
        require(::setitimer(ITIMER_REAL, &timer, nullptr) == 0, "setitimer");
        unsigned char value = 0;
        peer.recv_bytes(&value, 1);
        itimerval stop{};
        ::setitimer(ITIMER_REAL, &stop, nullptr);
        require(value == 42 && signalsSeen > 0, "EINTR interrupted a valid transfer");
    }, [](int fd) {
        std::this_thread::sleep_for(120ms);
        const unsigned char value = 42;
        rawSend(fd, &value, 1);
        ::close(fd);
    });

    stage = "partial EOF";
    socketPairTest([](int fd) {
        Peer3PC peer(fd, 1s);
        std::array<unsigned char, 16> data{};
        expectFailure([&] { peer.recv_bytes(data.data(), data.size()); });
        require(peer.bytesReceived() == 7, "partial EOF counters");
        require(::fcntl(fd, F_GETFD) == -1 && errno == EBADF, "failed connection not closed");
    }, [](int fd) {
        const std::array<unsigned char, 7> data{};
        rawSend(fd, data.data(), data.size());
        ::close(fd);
    });

    stage = "write after remote close / SIGPIPE";
    socketPairTest([](int fd) {
        Peer3PC peer(fd, 1s);
        pollfd event{fd, POLLIN, 0};
        require(::poll(&event, 1, 1000) > 0, "remote close notification");
        char byte;
        require(::recv(fd, &byte, 1, MSG_PEEK) == 0, "expected EOF before write");
        expectFailure([&] { peer.send_bytes("x", 1); });
    }, [](int fd) { ::close(fd); });

    stage = "idle I/O timeout";
    socketPairTest([](int fd) {
        Peer3PC peer(fd, 100ms);
        char byte;
        const auto start = std::chrono::steady_clock::now();
        bool timedOut = false;
        try { peer.recv_bytes(&byte, 1); }
        catch (const std::system_error& error) { timedOut = error.code() == std::errc::timed_out; }
        require(timedOut, "idle read did not time out");
        require(std::chrono::steady_clock::now() - start < 1s, "idle timeout took too long");
    }, [](int fd) { std::this_thread::sleep_for(300ms); ::close(fd); });

    stage = "absolute deadline despite progress";
    socketPairTest([](int fd) {
        Peer3PC peer(fd, 100ms);
        char data[20];
        const auto start = std::chrono::steady_clock::now();
        bool timedOut = false;
        try { peer.recv_bytes(data, sizeof(data)); }
        catch (const std::system_error& error) { timedOut = error.code() == std::errc::timed_out; }
        require(timedOut && peer.bytesReceived() > 0, "partial progress reset the timeout");
        require(std::chrono::steady_clock::now() - start < 400ms, "deadline was per-read, not per-operation");
    }, [](int fd) {
        for (int i = 0; i < 20; ++i) {
            std::this_thread::sleep_for(30ms);
            if (::send(fd, "x", 1, MSG_NOSIGNAL) < 0) break;
        }
        ::close(fd);
    });
}

int fdCount()
{
    DIR* directory = ::opendir("/proc/self/fd");
    require(directory != nullptr, "open fd directory");
    int count = 0;
    while (auto* entry = ::readdir(directory)) if (entry->d_name[0] != '.') ++count;
    ::closedir(directory);
    return count;
}

int connectRaw(const Endpoint3PC& endpoint)
{
    const auto deadline = std::chrono::steady_clock::now() + 3s;
    // Wait for the Linux listener before sending the single malformed fixture.
    // The test uses a directly routed loopback destination AND source address.
    for (;;) {
        std::ifstream table("/proc/net/tcp");
        std::string line;
        bool listening = false;
        while (std::getline(table, line)) {
            std::istringstream fields(line);
            std::string index, local, remote, state;
            if (!(fields >> index >> local >> remote >> state) || state != "0A") continue;
            const auto colon = local.find(':');
            if (colon != std::string::npos &&
                std::stoul(local.substr(colon + 1), nullptr, 16) == endpoint.port) listening = true;
        }
        if (listening) break;
        require(std::chrono::steady_clock::now() < deadline, "raw server listener timeout");
        std::this_thread::sleep_for(5ms);
    }
    for (;;) {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        require(fd >= 0, "raw TCP socket");
        sockaddr_in address{};
        address.sin_family = AF_INET;
        require(::inet_pton(AF_INET, endpoint.ip.c_str(), &address.sin_addr) == 1,
                "raw loopback address");
        address.sin_port = 0;
        if (::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
            ::close(fd);
            throw std::runtime_error("raw loopback source bind failed");
        }
        address.sin_port = htons(endpoint.port);
        if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0) return fd;
        ::close(fd);
        require(std::chrono::steady_clock::now() < deadline, "raw connect timeout");
        std::this_thread::sleep_for(10ms);
    }
}

// A distinct source address catches implementations that bind the destination
// address instead of the initiating party's configured loopback address.
void testLoopbackSourceAddress()
{
    stage = "configured loopback source address";
    Ports ports;
    auto endpoints = ports.endpoints;
    endpoints[2].ip = "127.0.0.3";
    runJobs({
        [=] {
            const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
            require(listener >= 0, "source regression listener");
            sockaddr_in address{};
            address.sin_family = AF_INET;
            address.sin_port = htons(endpoints[0].port);
            require(::inet_pton(AF_INET, endpoints[0].ip.c_str(), &address.sin_addr) == 1,
                    "source regression destination");
            require(::bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0,
                    "source regression bind");
            require(::listen(listener, 1) == 0, "source regression listen");
            pollfd event{listener, POLLIN, 0};
            require(::poll(&event, 1, 2500) > 0, "loopback connection did not reach listener");
            sockaddr_in remote{};
            socklen_t length = sizeof(remote);
            const int fd = ::accept(listener, reinterpret_cast<sockaddr*>(&remote), &length);
            require(fd >= 0, "source regression accept");
            in_addr expected{};
            require(::inet_pton(AF_INET, endpoints[2].ip.c_str(), &expected) == 1,
                    "source regression expected address");
            require(remote.sin_addr.s_addr == expected.s_addr,
                    "outgoing socket ignored the party's loopback source address");
            std::array<unsigned char, 8> hello{};
            rawRead(fd, hello.data(), hello.size());
            require(hello == std::array<unsigned char, 8>{{'R','S','S','3',1,0,2,0}},
                    "source regression hello");
            // Reject intentionally, so the client stops without requiring P1.
            const std::array<unsigned char, 8> badReply{{'R','S','S','3',2,0,0,2}};
            rawSend(fd, badReply.data(), badReply.size());
            ::close(fd);
            ::close(listener);
        },
        [=] {
            bool reachedServer = false;
            try { Comm3PC comm(party(2), endpoints, {3s, 1s}); }
            catch (const std::runtime_error& error) {
                reachedServer = dynamic_cast<const std::system_error*>(&error) == nullptr
                    && std::string(error.what()) == "RSS invalid protocol version or party handshake";
            }
            require(reachedServer, "loopback source regression failed before protocol validation");
        }
    }, ports.release(), true);
}

void testConstructionFailures()
{
    stage = "constructor validation";
    const std::array<Endpoint3PC, 3> example{{{"127.0.0.2", 43000},
                                           {"127.0.0.2", 43001}, {"127.0.0.2", 43002}}};
    const auto before = fdCount();
    expectInvalid([&] { Comm3PC c(party(-1), example); });
    expectInvalid([&] { Comm3PC c(party(3), example); });
    expectInvalid([&] { Comm3PC c(party(0), example, {0ms, 1s}); });
    expectInvalid([&] { Comm3PC c(party(0), example, {1s, 0ms}); });
    auto bad = example;
    bad[1].ip = "not-an-ip";
    expectInvalid([&] { Comm3PC c(party(0), bad); });
    bad = example; bad[0].port = 0;
    expectInvalid([&] { Comm3PC c(party(0), bad); });
    bad = example; bad[2] = bad[1];
    expectInvalid([&] { Comm3PC c(party(0), bad); });
    require(before == fdCount(), "invalid construction leaked descriptors");

    // Cover both accept timeout (P0) and repeated refused connects (P2).
    for (int own : {0, 2}) {
        stage = "absent parties " + std::to_string(own);
        Ports ports;
        const auto endpoints = ports.endpoints;
        runJobs({[=] {
            const auto baseline = fdCount();
            const auto start = std::chrono::steady_clock::now();
            bool timedOut = false;
            try { Comm3PC c(party(own), endpoints, {150ms, 150ms}); }
            catch (const std::system_error& error) {
                timedOut = error.code() == std::errc::timed_out;
                if (!timedOut) throw std::runtime_error(std::string("unexpected connection error: ") + error.what());
            }
            require(timedOut, "missing party did not cause connection timeout");
            require(std::chrono::steady_clock::now() - start < 1s, "connection deadline exceeded");
            require(baseline == fdCount(), "failed construction leaked descriptors");
        }}, ports.release(), true);
    }

    for (int mode = 0; mode < 5; ++mode) {
        stage = "bad/duplicate handshake " + std::to_string(mode);
        Ports ports;
        const auto endpoints = ports.endpoints;
        runJobs({
            [=] {
                const auto baseline = fdCount();
                bool rejected = false;
                try { Comm3PC c(party(0), endpoints, {2s, 1s}); }
                catch (const std::runtime_error& error) {
                    // A timeout is a failure of the test, not successful validation.
                    rejected = dynamic_cast<const std::system_error*>(&error) == nullptr;
                }
                require(rejected, "malformed handshake was not rejected immediately");
                require(baseline == fdCount(), "handshake rejection leaked descriptors");
            },
            [=] {
                const int fd = connectRaw(endpoints[0]);
                std::array<unsigned char, 8> hello{{'R', 'S', 'S', '3', 1, 0, 1, 0}};
                if (mode == 0) hello[0] = 'X';
                if (mode == 1) hello[4] = 2;
                if (mode == 2) hello[7] = 2;
                rawSend(fd, hello.data(), hello.size());
                if (mode >= 3) {
                    std::array<unsigned char, 8> reply{};
                    rawRead(fd, reply.data(), reply.size());
                    const unsigned char confirmation = mode == 3 ? 0xc3 : 0;
                    rawSend(fd, &confirmation, 1);
                    if (mode == 3) {
                        const int duplicate = connectRaw(endpoints[0]);
                        rawSend(duplicate, hello.data(), hello.size());
                        ::close(duplicate);
                    }
                }
                ::close(fd);
            }
        }, ports.release(), true);
    }

    stage = "retry a connected but silent startup endpoint";
    {
        Ports ports;
        const auto endpoints = ports.endpoints;
        std::vector<Job> jobs;
        for (int own = 0; own < 3; ++own) {
            jobs.push_back([=] {
                int stalled = -1;
                if (own == 0) {
                    // Accept one connection without replying. Release the listening
                    // port, but keep that connection open while the real P0 starts.
                    const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
                    require(listener >= 0, "stalled endpoint socket");
                    const int one = 1;
                    require(::setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) == 0,
                            "stalled endpoint reuse");
                    sockaddr_in address{};
                    address.sin_family = AF_INET;
                    address.sin_addr.s_addr = htonl(INADDR_ANY);
                    address.sin_port = htons(endpoints[0].port);
                    require(::bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0,
                            "stalled endpoint bind");
                    require(::listen(listener, 4) == 0, "stalled endpoint listen");
                    stalled = ::accept(listener, nullptr, nullptr);
                    require(stalled >= 0, "stalled endpoint accept");
                    ::close(listener);
                } else {
                    std::this_thread::sleep_for(50ms);
                }
                Comm3PC comm(party(own), endpoints, {30s, 500ms});
                if (stalled >= 0) ::close(stalled);
                comm.sync();
            });
        }
        runJobs(jobs, ports.release(), true);
    }

    stage = "party exits after mesh initialization";
    Ports ports;
    const auto endpoints = ports.endpoints;
    std::vector<Job> jobs;
    for (int own = 0; own < 3; ++own) {
        jobs.push_back([=] {
            Comm3PC comm(party(own), endpoints, {30s, 500ms});
            comm.sync();
            if (own == 0) ::_exit(0); // Deliberately skip C++ destruction.
            expectFailure([&] { comm.sync(); });
            comm.close(); comm.close();
        });
    }
    runJobs(jobs, ports.release(), true);
}
} // namespace

int main()
{
    try {
        testInvalidAndWire();
        testTransportFailures();
        testLoopbackSourceAddress();
        testConstructionFailures();
        std::array<int, 3> order{{0, 1, 2}};
        int runs = 0;
        do {
            stage = "startup order " + std::to_string(order[0])
                + std::to_string(order[1]) + std::to_string(order[2]);
            Ports ports;
            const auto endpoints = ports.endpoints;
            std::vector<Job> jobs;
            for (int own = 0; own < 3; ++own) {
                const int delay = static_cast<int>(std::find(order.begin(), order.end(), own)
                    - order.begin()) * 60;
                const bool full = runs == 0;
                jobs.push_back([=] { meshNode(own, endpoints, delay, full); });
            }
            runJobs(jobs, ports.release(), true);
            std::cout << "Passed " << stage << std::endl;
            ++runs;
        } while (std::next_permutation(order.begin(), order.end()));
        std::cout << "RSS TCP checks passed: 6 startup orders, widths 1..64, "
                     "all scalar/array types, 768 PRNG samples per party per order, "
                     "8 MiB rings, 4 MiB duplex exchanges, validation, framing fixtures, "
                     "EINTR, partial I/O, EOF, SIGPIPE, deadlines, loopback source binding and cleanup";
#ifdef NDEBUG
        std::cout << " (NDEBUG enabled)";
#endif
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RSS TCP test failed: " << error.what() << '\n';
        return 1;
    }
}
