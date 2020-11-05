/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package main;

/**
 *
 * @author jatawatsafe
 */
public class ClassificationProgram {
    public static void main(String[] args) {
        String trainFile = "creditRisk_Clean_NoCreditHistory_training.arff";
        String testFile = "creditRisk_Clean_NoCreditHistory_testing.arff";
        String predictFile = "creditRisk_Clean_NoCreditHistory_predicting.arff";
        //Step 1
        ML_Loan loan = new ML_Loan(9, trainFile, testFile, predictFile);
        loan.trainAndTestDataSet();
        // Step 2 predict
        loan.predictDataset();
        // 0 stands-for yes from @attribute Loan_Status {Y, N}
        

    }
}
