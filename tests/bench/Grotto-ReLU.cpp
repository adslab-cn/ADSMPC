#include <FSS/api.h>
#include <FSS/config.h>
#include <backend/FSS_transformer.h>
#include <module.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

GroupElement deterministicWord(GroupElement index, GroupElement salt)
{
    GroupElement z = index + salt + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

GroupElement clearInputAt(int index)
{
    if (index == 0)
        return 0;
    if (index == 1)
        return (GroupElement{1} << 63) - 1;
    if (index == 2)
        return GroupElement{1} << 63;
    if (index == 3)
        return GroupElement{0} - 1;

    GroupElement magnitude = deterministicWord(index, 11)
                           & ((GroupElement{1} << 40) - 1);
    return (index & 1) ? GroupElement{0} - magnitude : magnitude;
}

GroupElement plainReLU(GroupElement value, int bin)
{
    mod(value, bin);
    return (value >> (bin - 1)) & 1 ? 0 : value;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <party_id> [server_ip]\n"
                  << "  party_id: 1=dealer, 2=server, 3=client\n";
        return 1;
    }

    const int role = std::atoi(argv[1]);
    if (role < DEALER || role > CLIENT) {
        std::cerr << "party_id must be 1 (dealer), 2 (server), or 3 (client)\n";
        return 1;
    }
    const std::string ip = argc >= 3 ? argv[2] : "127.0.0.1";
    constexpr int bin = 64;
    constexpr int numSamples = 10000;

    sytorch_init();
    FSSConfig::bitlength = bin;
    FSSConfig::party = role;
    FSSConfig::num_threads = 4;

    auto *backend = new FSSTransformer<GroupElement>();
    backend->init(ip, true);

    std::vector<GroupElement> input(numSamples, 0);
    std::vector<GroupElement> output(numSamples, 0);
    std::vector<GroupElement> inputMask(numSamples, 0);
    std::vector<GroupElement> outputMask(numSamples, 0);
    std::vector<GroupElement> clearInput(numSamples, 0);

    for (int i = 0; i < numSamples; ++i) {
        const GroupElement x = clearInputAt(i);
        const GroupElement mask = deterministicWord(i, 37);
        clearInput[i] = x;
        if (role == DEALER)
            inputMask[i] = mask;
        else {
            input[i] = x + mask;
            mod(input[i], bin);
        }
    }

    FSS::start();
    const auto start = std::chrono::high_resolution_clock::now();
    GrottoReLU(numSamples, input.data(), output.data(), inputMask.data(),
                outputMask.data(), "Grotto-ReLU::");
    const auto end = std::chrono::high_resolution_clock::now();
    FSS::end();

    int exitCode = 0;
    if (role == DEALER) {
        FSSConfig::client->send_batched_input(outputMask.data(), numSamples, bin);
    } else if (role == CLIENT) {
        std::vector<GroupElement> receivedMask(numSamples);
        FSSConfig::dealer->recv_ge_array(receivedMask.data(), numSamples);

        int correct = 0;
        int reportedErrors = 0;
        for (int i = 0; i < numSamples; ++i) {
            const GroupElement result = output[i] - receivedMask[i];
            GroupElement normalizedResult = result;
            mod(normalizedResult, bin);
            const GroupElement expected = plainReLU(clearInput[i], bin);
            if (normalizedResult == expected) {
                ++correct;
            } else if (reportedErrors < 10) {
                std::cerr << "Mismatch at " << i
                          << ": expected=" << static_cast<int64_t>(expected)
                          << ", got=" << static_cast<int64_t>(normalizedResult)
                          << '\n';
                ++reportedErrors;
            }
        }
        std::cout << "Accuracy: " << correct << '/' << numSamples << '\n';
        std::cout << "Result: "
                  << (correct == numSamples ? "SUCCESS" : "FAILED") << '\n';
        exitCode = correct == numSamples ? 0 : 1;
    }

    std::cout << "Protocol Time="
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us\n";
    backend->finalize();
    delete backend;
    return exitCode;
}
