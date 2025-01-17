#include <iomanip>
#include <train.h>
#include <datasets/cifar10.h>
#include <datasets/mnist.h>
#include <layers/softmax.h>

void printprogress(double percent) {
    int val = (int) (percent * 100);
    int lpad = (int) (percent * 50);
    int rpad = 50 - lpad;
    std::cout << "\r" << "[" << std::setw(3) << val << "%] ["
              << std::setw(lpad) << std::setfill('=') << ""
              << std::setw(rpad) << std::setfill(' ') << "] ";
    std::cout.flush();
}
