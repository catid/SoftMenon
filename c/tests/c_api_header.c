/* Compile and link as C, so accidental C++ name mangling is a regression. */
#include "debayer.h"

int main(void) {
    return debayer_mirror_image(0, 0, 0, 0, 0) == cudaErrorInvalidValue ? 0 : 1;
}
