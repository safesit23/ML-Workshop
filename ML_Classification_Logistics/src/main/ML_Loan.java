/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package main;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.pmml.producer.LogisticProducerHelper;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
/**
 *
 * @author jatawatsafe
 */
public class ML_Loan {
    Classifier classifier; //Need to import weka library
    int classAttr;
    String trainFile, testFile, predictFile;
    String pmmlLoanModel;

    public ML_Loan(int classAttr, String trainFile, String testFile, String predictFile) {
        this.classAttr = classAttr;
        this.trainFile = trainFile;
        this.testFile = testFile;
        this.predictFile = predictFile;
    }
    
    //Instances - from weka (will return more than one)
    public Instances getDataFile(String fileName, int classifyAttr){
        try {
            int classIdx = classifyAttr;
            //read file arff by using weka lib
            ArffLoader loader = new ArffLoader();
            //to load file
            loader.setFile(new File(fileName));
            Instances dataSet = loader.getDataSet();
            //@data: Yes,1,NotGraduate,No,2653,1500,113,180,Rural,N
            dataSet.setClassIndex(classIdx); //classIdx = 9 is result that show can loan or not from data
            return dataSet;
        } catch (IOException ex) {
            Logger.getLogger(ML_Loan.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }
    
    public void trainAndTestDataSet(){
        try {
            Instances trainDataSet = getDataFile(trainFile, classAttr); //get from attr
            Instances testDataSet = getDataFile(testFile, classAttr);  //get from attr
            classifier = new Logistic();
            //send dataset to learn
            classifier.buildClassifier(trainDataSet);
            //evaluate after learning
            Evaluation eval = new Evaluation(trainDataSet);
            eval.evaluateModel(classifier, testDataSet);
            System.out.println("Logistics Evaluation");
            //summary
            System.out.println(eval.toSummaryString());
           
            //-------Part 2: Keep model to file
            System.out.println(classifier);
            Instances m_structure = new Instances(trainDataSet, 536);
            m_structure.setClassIndex(trainDataSet.numAttributes()-1);
            trainDataSet.setClassIndex(trainDataSet.numAttributes()-1);
            int m_NumClasses = trainDataSet.numClasses();
            int class_index = trainDataSet.classIndex();
            int nK = m_NumClasses-1;
            int nR = trainDataSet.numAttributes()-1;
            //2 dimension arr
            double[][] m_par = new double[nR+2][nK];
            pmmlLoanModel = LogisticProducerHelper.toPMML(trainDataSet, m_structure, m_par, m_NumClasses);
            
            String file = "loan_model";
            FileWriter fileW = new FileWriter(file);
            fileW.write(pmmlLoanModel);
            
        } catch (Exception ex) {
            Logger.getLogger(ML_Loan.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public void predictDataset(){
        System.out.println("Predicting");
        Instance predictDataSet;
        double value;
        Instances predictDataSets = getDataFile(predictFile, classAttr);
        // loop for instances
        for (int i = 0; i < predictDataSets.numInstances(); i++) {
            try {
                predictDataSet = predictDataSets.instance(i);
                // get result from predict file
                value = classifier.classifyInstance(predictDataSet);
                System.out.println("Allow to loan: "+((value==0)?"Yes":"No"));
            } catch (Exception ex) {
                Logger.getLogger(ML_Loan.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
    }
    
    
}
