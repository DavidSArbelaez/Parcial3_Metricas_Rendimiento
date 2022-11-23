/*********************************************************
 * Fecha: 21-09-2022
 * Autor: David Arbelaez
 * Materia: HPC-2
 * Tema: Implementacion del algoritmo de Regresion Logistica
 * Requerimientos:
 *      1.- Crear una clase que permita la manipulacion
 *          de kits de datos(extraccion, normalizacion,
 *          entre otros) con Eigen.
 *      2.- Crear una clase que permita implementar el
 *          modelo o algoritmo de Regresion Logistica con
 *          Eigen.
 * ******************************************************/
#include "Extraction/extraction.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <string.h>
#include "LogisticClass/logisticregression.h"

/* Principal/Main: Captura los argumentos de entrada
 * Lugar en donde se encuentra el dataset
 * Separador: Delimitador del dataset(;/,/./ /etc)
 * Header: Si tiene o no cabecera <se elimina si se tiene>*/
int main(int argc, char *argv[]){
    /* Se crea un objeto del tipo Extraer
     * para incluir los 3 argumentos que necesita
     * el objeto. */
    Extraction extraerData(argv[1], argv[2], argv[3]);

    /*std::cout << "Argumento 1: " << std::endl;
    std::cout << argv[1] << std::endl;
    std::cout << "Argumento 1: " << std::endl;
    std::cout << argv[2] << std::endl;
    std::cout << "Argumento 1: " << std::endl;
    std::cout << argv[3] << std::endl;*/

    /* Se requiere probar la lectura del fichero y
     * luego se requiere observar el dataset como
     * un objeto de matriz tipo dataframe. */
    std::vector<std::vector<std::string>> dataSET = extraerData.ReadCSV();
    int filas = dataSET.size()+1;
    int columnas = dataSET[0].size();
    Eigen::MatrixXd MatrizDataF = extraerData.CSVToEigen(
                 dataSET, filas, columnas);

    /* Se imprime la matriz que tiene los datos del
     * dataset. *///LogisticProject
    /*std::cout << "                      **** - Se imprime el Dataset - ****                     " << std::endl;
    std::cout << MatrizDataF << std::endl;
    std::cout << "Filas : " << filas << std::endl;
    std::cout << "Columnas: " << columnas << std::endl;*/

    /*std::cout << "Promedio: " << extraerData.Promedio(MatrizDataF) << std::endl;
    std::cout << "Desviacion: " << extraerData.DesvStandard(MatrizDataF) << std::endl;*/
    Eigen::MatrixXd matrizNorm = extraerData.Normalizacion(MatrizDataF, false);
    //std::cout << matrizNorm << std::endl;


    /* Se divide en 4 grupos entre los Test y Split para el procesamiento de los datos:
     *      X_Train
     *      y_Train
     *      X_Test
     *      y_Test */
    Eigen::MatrixXd X_Train, y_Train, X_Test, y_Test;
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> datos_divididos = extraerData.TrainTestSplit(matrizNorm, 0.8);

    /* datos_divididos presenta la tupla comprimida con 4 matrices. A continuacion se debe
     * descomprimir para las 4 matrices */
    std::tie(X_Train, y_Train, X_Test, y_Test) = datos_divididos;

    /* Inspección visual de columnas y filas en entrenamiento */
    /*std::cout << "Total de filas:                   " << matrizNorm.rows() << std::endl;
    std::cout << "Total de columnas:                " << matrizNorm.cols() << std::endl;
    std::cout << "===================================================================================================================" << std::endl;
    std::cout << "Total filas entrenamiento F:      " << X_Train.rows() << std::endl;
    std::cout << "Total columnas entrenamiento F:   " << X_Train.cols() << std::endl;
    std::cout << "Total filas entrenamiento T:      " << y_Train.rows() << std::endl;
    std::cout << "Total columnas entrenamiento T:   " << y_Train.cols() << std::endl;
    std::cout << "===================================================================================================================" << std::endl;
    std::cout << "Total filas prueba F:             " << X_Test.rows() << std::endl;
    std::cout << "Total columnas prueba F:          " << X_Test.cols() << std::endl;
    std::cout << "Total filas prueba T:             " << y_Test.rows() << std::endl;
    std::cout << "Total columnas prueba T:          " << y_Test.cols() << std::endl;*/

    /* A continuación se instacncia el objeto regresion lineal*/
    LogisticRegression modelo_lr ;

    /* Se ajustan los parametros */
    int dimension = X_Train.cols();
    Eigen::MatrixXd W = Eigen::VectorXd::Zero(dimension);
    double b=0;
    double lambda =0.0;
    bool log_costo = true;
    double learning_rate =0.01;
    int num_iter=10000;

    Eigen::MatrixXd dw;
    double db;
    std::vector<double> lista_costos;

    std::tuple<Eigen::MatrixXd, double,Eigen::MatrixXd,double, std::vector<double>> optimo = modelo_lr.Optimization(W, b, X_Train, y_Train, num_iter, learning_rate, lambda, log_costo);

    /* Se desempaqueta el conjunto de valores optimo */
    std::tie(W,b,dw,db,lista_costos)=optimo;



    /* Se crean las matrices de prediccion (de prueba y entrenamiento)*/
    Eigen::MatrixXd y_pred_test = modelo_lr.Prediction(W,b,X_Test);
    Eigen::MatrixXd y_pred_train = modelo_lr.Prediction(W,b,X_Train);

    auto prom=extraerData.Promedio(X_Train);
    auto dst=extraerData.DesvStandard(y_pred_train);
    //std::cout<<"Promedio de los datos de entrenamiento:"<<prom<<std::endl;
    //std::cout<<"Desviacion estandar de los datos de entrenamiento:"<<dst<<std::endl;

    /*std::cout<<y_pred_train.rows() <<std::endl;
    std::cout<<"****************************"<<std::endl;
    std::cout<<y_pred_test.rows() <<std::endl;*/

    /* A continuacion se calcula la metrica de accuracy para determinar la calidad del modelo*/
    /*auto train_Accuracy = 100-((y_pred_train - y_Train).cwiseAbs().mean() * 100);
    auto test_Accuracy = 100-((y_pred_test - y_Test).cwiseAbs().mean() * 100);

    std::cout<<"Accuracy de entrenamiento: "<<train_Accuracy<<"%"<<std::endl;
    std::cout<<"Accuracy de prueba: "<<test_Accuracy<<"%"<<std::endl;*/


    std::tuple<float,float,float,float> metricasTrain = extraerData.Metricas(y_pred_train,y_Train);
    std::tuple<float,float,float,float> metricasTest = extraerData.Metricas(y_pred_test,y_Test);
    float accuracy_train, precision_train, recall_train, f1_score_train,accuracy_test, precision_test, recall_test, f1_score_test;
    /* Se desempaqueta el conjunto de valores optimo */
    std::tie(accuracy_train, precision_train, recall_train, f1_score_train)=metricasTrain;
    std::tie(accuracy_test, precision_test, recall_test, f1_score_test)=metricasTest;

    std::cout<<"Accuracy de entrenamiento: "<<accuracy_train<<"%"<<std::endl;
    std::cout<<"Accuracy de prueba: "<<accuracy_test<<"%"<<std::endl;
    std::cout<<"precision de entrenamiento: "<<precision_train<<"%"<<std::endl;
    std::cout<<"precision de prueba: "<<precision_test<<"%"<<std::endl;
    std::cout<<"recall de entrenamiento: "<<recall_train<<"%"<<std::endl;
    std::cout<<"recall de prueba: "<<recall_test<<"%"<<std::endl;
    std::cout<<"f1_score de entrenamiento: "<<f1_score_train<<"%"<<std::endl;
    std::cout<<"f1_score de prueba: "<<f1_score_test<<"%"<<std::endl;
    /*Se guradan los datos de la lista de costos a un txt*/
    //extraerData.vector_to_file(lista_costos,"lista_costos.txt");
    return EXIT_SUCCESS;
}

