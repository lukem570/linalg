#include <linalg/linalg.hpp>

int main(void) {

    Linalg::Tensor<2, 3, 2> t;

    Linalg::Tensor<2, 2, 3> tP = t.permute(1, 2);

    tP *= 2;

    (void)tP;
}