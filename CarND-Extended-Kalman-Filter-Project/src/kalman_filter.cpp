#include <iostream>
#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
    /** from "Laser Measurements Part 4" Lesson */
    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    /** from "Laser Measurements Part 4" Lesson */
    VectorXd z_pred = H_ * x_;
  	VectorXd y = z - z_pred;
  	MatrixXd Ht = H_.transpose();
  	MatrixXd S = H_ * P_ * Ht + R_;
  	MatrixXd Si = S.inverse();
  	MatrixXd PHt = P_ * Ht;
  	MatrixXd K = PHt * Si;

  	//new estimate
  	x_ = x_ + (K * y);
  	long x_size = x_.size();
  	MatrixXd I = MatrixXd::Identity(x_size, x_size);
  	P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    /** from "EKF Algorithm Generalization" Lesson */
    /**
        p_dot_x = x_[0]
        p_dot_y = x_[1]
        v_dot_x = x_[2]
        v_dot_y = x_[3]
    */
    /**
                 _________
               /  2     2
    rho =   | / p'  + p'
            |/    x     y
    */
    float rho = sqrt(pow(x_[0],2) + pow(x_[1],2));

    /**
                    /p' \
                    |  y|
    phi =     arctan|---|
                    |p' |
                    \  x/
    */
    float phi = std::numeric_limits<float>::max();
    //check division by zero
  	if(fabs(x_[0]) < 0.0001){
        std::cout << "UpdateEKF () - Warning - Division by Zero, phi is max" << std::endl;
    } else {
        phi = atan2(x_[1],x_[0]);
    }

    /**
                p' v'  + p' v'
                  x  x     y  y
    rho_dot  =  ---------------
                      rho
    */
    float rho_dot = std::numeric_limits<float>::max();
    //check division by zero
  	if(fabs(rho) < 0.0001){
        std::cout << "UpdateEKF () - Warning - Division by Zero, rho_dot is max" << std::endl;
    } else {
        rho_dot = (x_[0]*x_[2] + x_[1]*x_[3]) / rho;
    }

    /**
              /  rho  \
              |       |
    h(x')  =  |  phi  |
              |       |
              \rho_dot/
    */
    /* z_pred = h(x') */
    VectorXd z_pred(3);
    z_pred << rho, phi, rho_dot;

    /** from "Laser Measurements Part 4" Lesson */
    VectorXd y = z - z_pred;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    //new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}
