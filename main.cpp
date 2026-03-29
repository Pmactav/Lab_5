#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <numeric>
#include <vector>
#include <cmath>
#include "functions.h"
#include <Eigen/Dense>

int main() {
    MatrixXd l = ReadDatatoMatrix("../coordsXY_2026.txt");
    VectorXd xhat(3);
    xhat << 1.75,
            1.5,
            0.20;
    cout << l.rows() << endl;
    cout << Misclosure(l, xhat).transpose() << endl;
    return 0;
}