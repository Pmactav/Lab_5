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
    MatrixXd coords = ReadDatatoMatrix("../coordsXY_2026.txt");
    int n  = coords.rows();
    VectorXd l(2*n);
    for (int i = 0; i < n; i++) {
        l(2*i)   = coords(i, 0);
        l(2*i+1) = coords(i, 1);}
    cout << "l size: " << l.size() << endl;
    cout << "l rows: " << l.rows() << endl;
    cout << "l cols: " << l.cols() << endl;

    VectorXd xhat(3);
    xhat << 1.75,
            1.5,
            0.20;
    //cout << l.rows() << endl;
    VectorXd w = Misclosure(l, xhat);
    MatrixXd A = AMatrix(l, xhat);
    MatrixXd B = BMatrix(l, xhat);
    double sigma = 0.005;
    MatrixXd Cl = MatrixXd::Identity(2*n,2*n)*sigma*sigma;
    double sigma0_sq = sigma*sigma;
    MatrixXd P = sigma0_sq*Cl.inverse();
    MatrixXd M = B*Cl*B.transpose();
    MatrixXd M_inv = M.inverse();
    MatrixXd N = A.transpose()*M_inv*A;
    MatrixXd u = A.transpose()*M_inv*w;
    VectorXd delta = -N.ldlt().solve(u);
    xhat += delta; //technichaly x0 for first iteration
    cout << "xc: " << xhat(0) << endl;
    cout << "yc: " << xhat(1) << endl;
    cout << "r:  " << xhat(2) << endl;
    VectorXd k = M_inv*(A*delta+w);
    VectorXd v_hat = -P.inverse()*B.transpose()*k;
    //cout << v_hat << endl;
    VectorXd l_hat = l + v_hat;
    cout << "sample distances:\n";
    for (int i = 0; i < 5; i++) {
        double xi = l(2*i);
        double yi = l(2*i+1);
        double d = sqrt(pow(xi - xhat(0),2) + pow(yi - xhat(1),2));
        cout << d << endl;
    }
    //cout << l_hat << endl;
    return 0;
}