#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <list>

class LogisticRegression
{
public:
    Eigen::MatrixXd Sigmoid(Eigen::MatrixXd Z);
    std::tuple<Eigen::MatrixXd, double, double> Propagation(Eigen::MatrixXd W, Eigen::MatrixXd X, double b, Eigen::MatrixXd y, double lambda);
    std::tuple<Eigen::MatrixXd, double, Eigen::MatrixXd, double,std::vector<double>> Optimization(Eigen::MatrixXd W, double b, Eigen::MatrixXd X, Eigen::MatrixXd y, int num_iter, double lr, double lambda, bool log_costo);
    Eigen::MatrixXd Prediction(Eigen::MatrixXd W, double b, Eigen::MatrixXd X);
};
#endif // LOGISTICREGRESSION_H
