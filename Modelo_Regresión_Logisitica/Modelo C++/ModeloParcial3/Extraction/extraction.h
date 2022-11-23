#ifndef EXTRACTION_H
#define EXTRACTION_H

#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <vector>

/* La clase extraction se compone de las funciones
 * o metodos para manipular el dataset.
 * Se presentan funciones para:
 * -- lectura csv
 * -- Promedios
 * -- Normalizacion de datos
 * -- Desviacion estandar
 * -- Funcion de perdida
 * La clase recibe como parametros de entrada
 * -- El dataset (path del dataset ".csv")
 * -- El delimitador (separador entre columnas del dataset)
 * -- Si el dataset tiene o no Cabecera (header) */

class Extraction
{
    /* Se presenta el constructor para los argumentos
     * de entrada a la clase: nombre_dataset, delimitador, header */

    /* Nombre del dataset */
    std::string setDatos;

    /* Separador de columnas */
    std::string delimitador;

    /* Si el dataset tiene cabecera o no*/
    bool header;

public:
    Extraction(std::string datos,
            std::string separador,
            bool head):
                    setDatos(datos),
                    delimitador(separador),
                    header(head){}
    /********** Prototipo de funciones propias de la clase **********/
    //Cabecera de funcion ReadCSV
    std::vector<std::vector<std::string>> ReadCSV();
    //Cabecera de funcion CSVToEigen
    Eigen::MatrixXd CSVToEigen(
            std::vector<std::vector<std::string>>  SETdatos,
            int filas, int columnas);
    //Cabecera de funcion Promedio
    auto Promedio(Eigen::MatrixXd datos) -> decltype(datos.colwise().mean());
    auto DesvStandard(Eigen::MatrixXd datos) -> decltype((datos.array().square().colwise().sum() / (datos.rows()-1)).sqrt());
    Eigen::MatrixXd Normalizacion(Eigen::MatrixXd datos, bool normalTarget);
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> TrainTestSplit(Eigen::MatrixXd data, float train_size);
    void vector_to_file(std::vector<double> vector, std::string nombre_file);
    void eigen_to_file(Eigen::MatrixXd datos, std::string nombre_file);
    std::tuple<float,float,float,float> Metricas(Eigen::MatrixXd Predict,Eigen::MatrixXd real);
};

#endif // EXTRACTION_H
