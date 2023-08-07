#include <iostream>
#include "gas/gas.h"

int main() {
    std::cout << "Hello, world!" << std::endl;
    GasStates gs = GasStates(5);
    std::cout << "Built some gas states!" << std::endl;
    return 0;
}
