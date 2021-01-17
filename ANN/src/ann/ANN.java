/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ann;

/**
 *
 * @author alaaf
 */
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;
import java.util.Scanner;
import java.util.Vector;

public class ANN {

    static FileWriter Fileoutput;

    public static double[] MultipleMatrixbyVector(double[][] matrix, double[] vector) {
        double[] newvec = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                newvec[i] += matrix[i][j] * vector[j];
            }
        }
        return newvec;
    }

    public static double[] feedForwardForHidden(double WeightsOfHiddenlayer[][], int L, double Xinputs[]) {
        double I[] = new double[WeightsOfHiddenlayer.length];
        double[] Neth = MultipleMatrixbyVector(WeightsOfHiddenlayer, Xinputs);
        for (int i = 0; i < L; i++) {
            I[i] = sigmoidactivation(Neth[i]);
        }
        return I;
    }

    public static double[] feedForwardForoutput(double WeightsOfoutputlayer[][], int L, int N, double I[]) {
        double a[] = new double[WeightsOfoutputlayer.length];
        double[] Neth = MultipleMatrixbyVector(WeightsOfoutputlayer, I);
        for (int i = 0; i < Neth.length; i++) {
            a[i] = sigmoidactivation(Neth[i]);
        }
        return a;
    }

    public static double MSE_(double Yactual[], double YPredicted[], int N) {
        double MSE_ = 0;
        for (int i = 0; i < N; i++) {
            MSE_ += Math.pow(Yactual[i] - YPredicted[i], 2);
        }
        return MSE_;
    }

    public static double[] CalculateErrorforLayer(double Youtputs[], int N, double a[]) {
        double error[] = new double[N];
        for (int i = 0; i < N; i++) {
            error[i] = a[i] - Youtputs[i];
        }
        return error;
    }

    public static double[] CalculateErrorforoutputLayer(double error[], int N, double a[]) {
        double DO[] = new double[N];
        for (int i = 0; i < N; i++) {
            DO[i] = a[i] * (1 - a[i]) * error[i];
        }
        return DO;
    }

    public static double[] CalculateErrorforhiddenLayer(double error[], int N, int L, double I[], double WeightsOfhiddenLayer[][]) {
        double DH[] = new double[L];
        for (int i = 0; i < L; i++) {
            double sum = 0;
            for (int j = 0; j < N; j++) {
                sum += error[j] * WeightsOfhiddenLayer[i][j];
            }
            DH[i] = I[i] * (1 - I[i]) * sum;
        }
        return DH;
    }

    // Change all weight values of each weight matrix using the formula
    // weight(old) + learning rate * output error * output(neurons i) *
    // output(neurons i+1) * ( 1 - output(neurons i+1) )
    public static double[][] UpdateWeightsforhiddenLayer(double wH[][], double learningrate, double x[], double DH[], int M,
            int L) throws IOException {
        Fileoutput.write("--------------------------------------------------------------- " + "\n");

        for (int i = 0; i < L; i++) {
            Fileoutput.write("UpdateWeightsforhiddenLayer for layer =" + (i + 1) + "\n");
            for (int j = 0; j < M; j++) {
                wH[i][j] = wH[i][j] - learningrate * DH[i] * x[j];
                Fileoutput.write("UpdateWeightsforhiddenLayer = " + wH[i][j] + "\n");
            }
        }
        return wH;

    }

    public static double[][] UpdateWeightsforoutputLayer(double wO[][], double learningrate, double I[], double DO[], int N,
            int L) throws IOException {
        Fileoutput.write("------------------------------------------------------------------------ UpdateWeightsforoutputLayer ----------------------------------------------------------------" + "\n");
        for (int i = 0; i < wO.length; i++) {
            for (int j = 0; j < wO[i].length; j++) {
                wO[i][j] = wO[i][j] - learningrate * DO[i] * I[j];
                Fileoutput.write("UpdateWeightsforoutputLayer = " + wO[i][j] + "\n");
            }
        }
        return wO;

    }

    // sigmoid activation
    public static double sigmoidactivation(double x) {
        return (1.0 / (1 + Math.exp(-x)));
    }

    // initialize weights
    public static void initializeWeights(double weight[][], int l1, int l2, int rangeMin, int rangeMax) {
        Random r = new Random();
        for (int i = 0; i < l1; i++) {
            for (int j = 0; j < l2; j++) {

                weight[i][j] = rangeMin + (rangeMax - rangeMin) * r.nextDouble();
            }
        }
    }

    // *******************************Normalization step
    // functions***************************************************************//
    public static double CalculateMean(double[] Xinputs, int M) {
        double mean = 0.0;
        for (int j = 0; j < M; j++) {
            mean += Xinputs[j];
        }
        mean /= M;
        return mean;
    }

    public static double CalculateStandardDiv(double[] Xinputs, int M, double mean) {
        double std_dev = 0.0;
        for (int j = 0; j < M; j++) {
            std_dev += Math.pow(Xinputs[j] - mean, 2);
        }
        std_dev = Math.sqrt(std_dev / 4);
        return std_dev;
    }

    public static double[] Gaussian_normalization(double[] Xinputs, int M, double mean, double std_dev) {
        double normalizedInputs[] = new double[M];
        for (int j = 0; j < M; j++) {
            normalizedInputs[j] = (Xinputs[j] - mean) / std_dev;
        }
        return normalizedInputs;
    }

    // ***********************************************************************************************//
    public static void main(String[] args) throws IOException {
        // TODO Auto-generated method stub

        int M, L, N; // m is the number of nodes , l is number of hidden nodes , N is the number of
        // outputs nodes ;
        int K; // number of training examples
        // pass the path to the file as a parameter
        Fileoutput = new FileWriter("C:\\Users\\alaaf\\OneDrive\\Documents\\NetBeansProjects\\ANN\\src\\ann\\output.txt");
        FileWriter NormalizedFile = new FileWriter("C:\\Users\\alaaf\\OneDrive\\Documents\\NetBeansProjects\\ANN\\src\\ann\\normalized.txt");

        File file = new File("C:\\Users\\alaaf\\OneDrive\\Documents\\NetBeansProjects\\ANN\\src\\ann\\trainingdata.txt");
        Scanner scan = new Scanner(file);
        M = scan.nextInt();
        System.out.println("Number of nodes = " + M);
        L = scan.nextInt();
        System.out.println("Number of hidden nodes = " + L);
        N = scan.nextInt();
        System.out.println("Number of outputs nodes = " + N);
        K = scan.nextInt();
        System.out.println("Number of Training Examples = " + K);
        double Xinputs[] = new double[M];
        double Youtputs[] = new double[N];
        double WeightsOfhiddenLayer[][] = new double[L][M];
        double WeightsOfoutputlayer[][] = new double[N][L];
        double updateWeightsHL[][] = new double[L][M];
        double updateWeightsOL[][] = new double[N][L];
        double I[] = new double[L];
        double A[] = new double[N];
        double error[] = new double[N];
        double DH[] = new double[L];
        double DO[] = new double[N];
        int numberofierations = 3;
        double MeanSquareerror = 0;
        double mean;
        double StandardDiv;
        double learningrate = 0.1;
        double[] Normalized = new double[M];
        // initialize Weights for hidden layer and output layer .
        initializeWeights(WeightsOfhiddenLayer, L, M, -2, 2);
        initializeWeights(WeightsOfoutputlayer, N, L, -2, 2);
        NormalizedFile.write("Number of nodes = " + M + "\n");
        NormalizedFile.write("Number of hidden nodes = " + L + "\n");
        NormalizedFile.write("Number of outputs nodes = " + N + "\n");
        NormalizedFile.write("Number of Training Examples = " + K + "\n");
        NormalizedFile.write("--------------------------------------------------" + "\n");
        NormalizedFile.write("WeightsOfhiddenLayer = " + "\n");
        // System.out.println("--------------------WeightsOfhiddenLayer--------------------");
        Vector<double[]> v = new Vector<double[]>();
        /* reading from file X inputs and y output */
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < M; j++) {
                Xinputs[j] = scan.nextDouble();
            }
            for (int j = 0; j < N; j++) {
                Youtputs[j] = scan.nextDouble();
            }
            mean = CalculateMean(Xinputs, M);
            StandardDiv = CalculateStandardDiv(Xinputs, M, mean);
            Normalized = Gaussian_normalization(Xinputs, M, mean, StandardDiv);
            for (int j = 0; j < M; j++) {
                NormalizedFile.write("Normalized Input " + (i + 1) + " = " + Normalized[j] + "\n");
            }
            v.add(i, Normalized);

        }
        double MSerors[] = new double[K];
        double min = 2000.0;
        for (int i = 0; i < numberofierations; i++) {
            for (int j = 0; j < K; j++) {

                I = feedForwardForHidden(WeightsOfhiddenLayer, L, v.get(j));
                A = feedForwardForoutput(WeightsOfoutputlayer, L, N, I);
                error = CalculateErrorforLayer(Youtputs, N, A);
                DO = CalculateErrorforoutputLayer(error, N, A);
                DH = CalculateErrorforhiddenLayer(DO, N, L, I, WeightsOfhiddenLayer);
                updateWeightsHL = UpdateWeightsforhiddenLayer(WeightsOfhiddenLayer, learningrate, v.get(j), DH, M, L);
                updateWeightsOL = UpdateWeightsforoutputLayer(WeightsOfoutputlayer, learningrate, I, DO, N, L);
                MeanSquareerror = MSE_(Youtputs, A, N);
                Fileoutput.write("MeanSquareerror = " + MeanSquareerror + "\n");
                MeanSquareerror = MSE_(Youtputs, A, N);
             //   Fileoutput.write("MeanSquareerror = " + MeanSquareerror + "\n");
                MSerors[j] = MeanSquareerror;
            }
        }
        for (int i = 0; i < K; i++) {
            if (MSerors[i] < min) {
                min = MSerors[i];
            }
        }

        Fileoutput.close();

        NormalizedFile.close();
    }

}
