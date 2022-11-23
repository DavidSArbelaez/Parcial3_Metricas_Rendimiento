#include "logisticregression.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <list>

/*Primera funcion miembro: la funcion sigmoid*/

Eigen::MatrixXd LogisticRegression::Sigmoid(Eigen::MatrixXd Z){
// Funcion sigmoid que retorna la matriz con el calculo de la funcion:

    return 1/(1+(-Z.array()).exp());
}

/*Segunda funcion: La funcion de propagación, la cual contiene el tratamiento
 * de la funcion de costo (cross entropia), y sus correspondientes derivadas*/

std::tuple<Eigen::MatrixXd, double, double> LogisticRegression::Propagation(Eigen::MatrixXd W, Eigen::MatrixXd X, double b,
                                                                            Eigen::MatrixXd y, double lambda){
    /*Sobre la base de la presentación de regresion logistica*/
    int m = y.rows();
    /*Se obtiene la matriz eigen Z*/
    Eigen::MatrixXd Z = (W.transpose() * X.transpose()).array()+b;
    Eigen::MatrixXd A = Sigmoid(Z);
    /*Se crea la funcion para la entropia cruzada, no se sabe que valor se va a retornar
     */
    auto cross_entropy = -(y.transpose()*(Eigen::VectorXd)A.array().log().transpose() + ((Eigen::VectorXd)(1-y.array())).transpose()*(Eigen::VectorXd)(1-A.array()).log().transpose())/m;

    /*Funcion para reducir la varianza del modelo usando la regularización*/


   double l2_reg_costo = W.array().pow(2).sum()*(lambda/(2*m));


   /*Función para el calculo del costo usando la entropía cruzada con el ajuste de la regularización*/
   /*Se hace uso de static_cast debido a que la funcion debe retornar un double pero va a estar compuesta de tipos de datos, definidos por el usuario
    *(auto)*/

//   double costo = static_cast<const double>(cross_entropy.array()[0]) + l2_reg_costo;
   double costo = static_cast<const double>((cross_entropy.array()[0]) + l2_reg_costo);
   /*Se calcula la derivada de las matrices en función de los pesos*/

   Eigen::MatrixXd dw = ((Eigen::MatrixXd) (A-y.transpose())*X/m)+(Eigen::MatrixXd)(lambda/m*W).transpose();

   /* Se calcula la derivada en función del bias(punto de corte) */
   double db = (A-y.transpose()).array().sum()/m;
   /*Se retorna de la función de propagación la derivada de los pesos(w), la derivada del bias y retornamos
    * el costo: Es una tupla comprimida*/
   return std::make_tuple(dw,db,costo);

}

/* Se crea la funcion de optimización: Se crea una lista en donde se va a almacenar cada uno de los valores de la funcion de costo hasta
 * converger. Esta actualización se almacenará en un fichero para posteriormente ser visualizada,
 * la actualizacion se observa en la presentacion de regresion logistica. */

std::tuple<Eigen::MatrixXd, double, Eigen::MatrixXd, double,std::vector<double>> LogisticRegression::Optimization(Eigen::MatrixXd W,
                                                                                                                double b, Eigen::MatrixXd X,
                                                                                                                Eigen::MatrixXd y,
                                                                                                                int num_iter, double lr,
                                                                                                                double lambda, bool log_costo){
    /*Se crea la lista*/
    std::vector<double> lista_costo;
    Eigen::MatrixXd dw;
    double db, costo;


    /*Se hace la iteración*/
    for(int i=0;i<num_iter;i++){
        std::tuple<Eigen::MatrixXd, double, double> propagate = Propagation( W, X,  b, y, lambda);
        std::tie(dw,db,costo) = propagate;

    //Se actualizan los valores que reprsentan los weights y biases,
        W = W - (lr*dw).transpose();
        b = b - (lr*db);

        /*según la bandera se guarda cada 100 pasos el valor del costo */
        if(i%100==0){
            lista_costo.push_back(costo);
        }

        //Se imprime por pantalla según la bandera
        /*if(log_costo && i % 100==0){
               std::cout<<"Costo después de actualizar "<< i << ": " << costo <<std::endl;
        }*/

}

    return std::make_tuple(W,b,dw,db,lista_costo);
}

/*La funcióm estimará la etiqueta de salida si corresponde a cero o uno, la idea es calcular ŷ (y estimado)
 * usando los parametros de regersion aprendidos, se convierten las entradas a 0 si la función es inferior o igual a 0.5.
 * Se convierten las entradas a 1 si es superior a 0.5.*/

Eigen::MatrixXd LogisticRegression::Prediction(Eigen::MatrixXd W, double b, Eigen::MatrixXd X){
/*Se calcula la cantidad de variables registros (m) */
    int m = X.rows();
    /*Creamos una matriz con valores del vector de ceros del tamaño de la matriz de entrada(X)*/
    Eigen::MatrixXd y_pred =Eigen::VectorXd::Zero(m).transpose();

    /*Se crea una matriz para almacenar los valores de z (calculados)*/

    Eigen::MatrixXd Z = (W.transpose()*X.transpose()).array() + b;

    /*Se calcula la función sigmoid en la matriz A */
    Eigen::MatrixXd A = Sigmoid(Z);

    /*Se calcula el valor estimado (Etiqutas 0 o 1) Segun la función de activación
     * para cada uno de los regsitros (matriz X)*/




    for (int i=0;i<A.cols(); i++){
        if (A(0,i)<=0.5){
            y_pred(0,i) = 0;
        }else{
            y_pred(0,i) = 1;
        }

    }
    return y_pred.transpose();



}


