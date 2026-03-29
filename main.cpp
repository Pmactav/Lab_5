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
    VectorXd w = Misclosure(l, xhat);
    MatrixXd A = AMatrix(l, xhat);
    MatrixXd B = BMatrix(l, xhat);
    double n  = l.rows();
    double sigma = 0.005;
    MatrixXd P = MatrixXd::Identity(2*n,2*n)*(1/(sigma*sigma));
    MatrixXd M = B * P.inverse() * B.transpose();
    MatrixXd N = A * P.inverse() * A.transpose();
    MatrixXd u = M * w;


    return 0;
}