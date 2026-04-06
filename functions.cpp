//
// Created by Peter on 2/4/2026.
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <vector>
#include <Eigen/Dense>
#include "functions.h"

using namespace std;
using namespace Eigen;

MatrixXd ReadDatatoMatrix(const std::string &filename) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "ERROR: Could not open file " << filename << "\n";
        return MatrixXd(0,0);}
    vector<vector<double>> data;
    string line;
    while (getline(infile, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        vector<double> row;
        double value;
        while (ss >> value) {row.push_back(value);}
        if (!row.empty())data.push_back(row);}
    if (data.empty())
        return MatrixXd(0,0);
    int rows = data.size();
    int cols = data[0].size();
    MatrixXd M(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            M(i,j) = data[i][j];
    return M;
}

void WriteMatrixToFile(const MatrixXd &Mat, const string& filename, unsigned int precision){
    ofstream out(filename, ios::out);
    if (out.fail()){
        cout << "Could not open output file " << filename << endl;
        exit(1);
    }
    out << fixed << setprecision(precision);
    for (int i = 0; i < Mat.rows(); ++i){
        for (int j = 0; j < Mat.cols(); ++j){
            out << Mat(i, j);
            if (j != Mat.cols() - 1)
                out << " ";
        }
        out << endl;
    }
    out.close();
}

VectorXd Misclosure(const VectorXd &l, const VectorXd &xhat) {
    int n = l.rows() / 2;
    VectorXd w(n);
    double x0 = xhat(0);
    double y0 = xhat(1);
    double r  = xhat(2);
    for (int i = 0; i < n; i++) {
        double xi = l(2*i);
        double yi = l(2*i+1);
        w(i) = pow(xi-x0,2)+pow(yi-y0,2)-r*r;
    }
    return w;
}

MatrixXd AMatrix(const VectorXd &l, const VectorXd &x_hat) {
    int n = l.rows() / 2;
    MatrixXd A(n, 3);
    double x0 = x_hat(0);
    double y0 = x_hat(1);
    double r0 = x_hat(2);
    for (int i = 0; i < n; i++) {
        double xi = l(2*i);
        double yi = l(2*i+1);
        A(i, 0) = -2*(xi - x0);
        A(i, 1) = -2*(yi - y0);
        A(i, 2) = -2*r0;
    }
    return A;
}

MatrixXd BMatrix(const VectorXd &l, const VectorXd &x_hat) {
    int n = l.rows() / 2;
    MatrixXd B = MatrixXd::Zero(n, 2*n);
    double x0 = x_hat(0);
    double y0 = x_hat(1);
    for (int i = 0; i < n; i++) {
        double xi = l(2*i);
        double yi = l(2*i+1);
        B(i, 2*i)   =  -2*(xi - x0);
        B(i, 2*i+1) =  -2*(yi - y0);
    }
    return B;
}


