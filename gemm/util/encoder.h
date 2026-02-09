#include <cstdint>
#include <utility>

namespace encoder {
void baseEncodeTern(int* A, uint8_t* A0, uint8_t* A1, int n, int m);

void baseEncodeBin(int* B, uint8_t* Bnew, int n, int m);

std::pair<uint8_t, uint8_t> addMul(int& c, uint8_t a0, uint8_t a1, uint8_t b);
};