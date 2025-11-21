#include <linalg/linalg.hpp>

int main(void) {

    Linalg::Tensor<2, 3, 2> t;

    Linalg::Tensor<2, 2, 3> tP = t.permute(1, 2);

    tP *= 2;

    la::Vec2 v1 = {1, 0};
    la::Vec2 v2 = 1;
    v2.x = 2;

    float det = v1.determinant(v2);

    printf("det: %f\n", det);
}