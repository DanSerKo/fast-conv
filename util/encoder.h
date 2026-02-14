#include <cstdint>
#include <utility>

namespace encoder {
void encodeTern(int* A, uint8_t* Anew, int n, int m);
void encodeBin(int* B, uint8_t* Bnew, int n, int m);

void addMulSingle(int& c, uint8_t a0, uint8_t a1, uint8_t b);
void addMul(int& c, uint8_t a0, uint8_t a1, uint8_t b);
};
