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
    //estimate x0
    int n  = coords.rows();
    double xmean = coords.col(0).mean();
    double ymean = coords.col(1).mean();
    double xrange = coords.col(0).maxCoeff()-(coords.col(0).minCoeff());
    double yrange = coords.col(1).maxCoeff()-(coords.col(1).minCoeff());
    double r_estimate = (xrange+yrange)/4;//average range and divide by two for radius /2/2 = /4
    //split x and y into single vector
    VectorXd l(2*n);
    for (int i = 0; i < n; i++) {
        l(2*i)   = coords(i, 0);
        l(2*i+1) = coords(i, 1);}
    VectorXd xhat(3);
    xhat << xmean,
            ymean,
            r_estimate;
    double sigma = 0.005;
    double sigma0_sq = sigma*sigma;
    MatrixXd Cl = MatrixXd::Identity(2*n,2*n)*sigma0_sq;
    MatrixXd P = Cl.inverse();
    //Begin iteration
    VectorXd delta;
    int iter = 0;
    do {
        VectorXd w = Misclosure(l, xhat);
        MatrixXd A = AMatrix(l, xhat);
        MatrixXd B = BMatrix(l, xhat);
        MatrixXd M = B * Cl * B.transpose();
        MatrixXd M_inv = M.inverse();
        MatrixXd N = A.transpose() * M_inv * A;
        MatrixXd u = A.transpose() * M_inv * w;
        delta = -N.ldlt().solve(u);
        xhat += delta;
        iter++;
    } while (delta.norm() > 1e-10 && iter < 20);
    cout << "Converged in " << iter << " iterations" << endl;
    WriteMatrixToFile(xhat, "../xhat.txt", 10);
    //rebuild outside of loop with final xhat
    VectorXd w = Misclosure(l, xhat);
    MatrixXd A = AMatrix(l, xhat);
    MatrixXd B = BMatrix(l, xhat);
    MatrixXd M = B * Cl * B.transpose();
    MatrixXd M_inv = M.inverse();
    VectorXd k = M_inv*(A*delta+w);
    VectorXd v_hat = -Cl*B.transpose()*k;
    MatrixXd N = A.transpose() * M_inv * A;
    MatrixXd u = A.transpose() * M_inv * w;
    VectorXd l_hat = l + v_hat;
    //calculate deviations
    int r = 2*n-3;
    MatrixXd N_inv = N.inverse();
    double sigma0_square_hat = (v_hat.transpose()*P*v_hat)(0,0)/r;
    MatrixXd C_xhat = sigma0_square_hat*N_inv;
    MatrixXd C_vhat = (Cl*B.transpose()*M_inv*B*Cl)-(Cl*B.transpose()*M_inv*A* N.inverse()*A.transpose()*M_inv*B*Cl);
    MatrixXd C_lhat = Cl-C_vhat;
    cout << "sigma0_hat:    " << sqrt(sigma0_square_hat) << endl;
    WriteMatrixToFile(C_xhat.diagonal().cwiseAbs().cwiseSqrt(), "../C_xhatdiag.txt", 10);
    WriteMatrixToFile(C_vhat.diagonal().cwiseAbs().cwiseSqrt(), "../C_vhatdiag.txt", 10);
    WriteMatrixToFile(C_lhat.diagonal().cwiseAbs().cwiseSqrt(), "../C_lhatdiag.txt", 10);
    cout << "Diagonals written to .txt" << endl;
    //conduct check
    VectorXd check = A*delta + B*v_hat + w;
    cout << "Check: " << check.norm() << endl;
    //Build Variance Covariance
    VectorXd std_xhat = C_xhat.diagonal().cwiseAbs().cwiseSqrt();
    MatrixXd D = std_xhat.asDiagonal().inverse();
    MatrixXd R_xhat = D * C_xhat * D;
    cout << "R_xhat: " << endl << R_xhat << endl;
    //split vectors back into x,y
    MatrixXd lhat_coords  = VectorToMatrix(l_hat, n);
    WriteMatrixToFile(lhat_coords, "../lhat_coords.txt", 10);
    MatrixXd vhat_coords  = VectorToMatrix(v_hat, n);
    WriteMatrixToFile(vhat_coords, "../vhat_coords.txt", 10);
    return 0;
}