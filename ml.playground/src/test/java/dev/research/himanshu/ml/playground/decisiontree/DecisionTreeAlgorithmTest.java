package dev.research.himanshu.ml.playground.decisiontree;

import java.math.BigDecimal;
import java.util.List;

import dev.research.himanshu.ml.playground.decisiontree.model.Instances;
import dev.research.himanshu.ml.playground.decisiontree.util.Utility;
import junit.framework.TestCase;

public class DecisionTreeAlgorithmTest extends TestCase {

	public void testDecisionTreeWithoutVariance() throws Exception {
		DecisionTreeAlgorithm dta = new DecisionTreeAlgorithm();
		dta.train(System.getProperty("user.dir") + "/src/main/resources/data_sets_1/training_set.csv");
		dta.generateDecisionTree();
		dta.printDecisionTree();
	}
	
	public void testDecisionTreeWithVariance() throws Exception {
		DecisionTreeAlgorithm dta = new DecisionTreeAlgorithm();
		dta.train(System.getProperty("user.dir") + "/src/main/resources/data_sets_1/training_set.csv", true);
		dta.generateDecisionTree(true);
		dta.printDecisionTree();
	}
	
	public void testDecisionTreeWithTrainingDataWithoutVariance() throws Exception {
		DecisionTreeAlgorithm dta = new DecisionTreeAlgorithm();
		dta.train(System.getProperty("user.dir") + "/src/main/resources/data_sets_1/training_set.csv");
		dta.generateDecisionTree();
		dta.printDecisionTree();
	
		List<String> validationDataLines = Utility.loadFile(System.getProperty("user.dir") + "/src/main/resources/data_sets_1/training_set.csv");
		Instances validationInstances = Utility.loadInstancesFromData(validationDataLines);
		BigDecimal accuracy = dta.validateInstances(validationInstances);
		
		System.out.println(accuracy);
	}
	
	public void testDecisionTreeWithTrainingDataWithVariance() throws Exception {
		DecisionTreeAlgorithm dta = new DecisionTreeAlgorithm();
		dta.train(System.getProperty("user.dir") + "/src/main/resources/data_sets_1/training_set.csv", true);
		dta.generateDecisionTree(true);
		dta.printDecisionTree();
	
		List<String> validationDataLines = Utility.loadFile(System.getProperty("user.dir") + "/src/main/resources/data_sets_1/training_set.csv");
		Instances validationInstances = Utility.loadInstancesFromData(validationDataLines);
		BigDecimal accuracy = dta.validateInstances(validationInstances);
		
		System.out.println(accuracy);
	}
	
	public void testDecisionTreeWithValidationDataWithoutVariance() throws Exception {
		DecisionTreeAlgorithm dta = new DecisionTreeAlgorithm();
		dta.train(System.getProperty("user.dir") + "/src/main/resources/data_sets_1/training_set.csv");
		dta.generateDecisionTree();
		dta.printDecisionTree();
		
		List<String> validationDataLines = Utility.loadFile(System.getProperty("user.dir") + "/src/main/resources/data_sets_1/validation_set.csv");
		Instances validationInstances = Utility.loadInstancesFromData(validationDataLines);
		BigDecimal accuracy = dta.validateInstances(validationInstances);
		
		System.out.println(accuracy);
	}
	
	public void testDecisionTreeWithValidationDataWithVariance() throws Exception {
		DecisionTreeAlgorithm dta = new DecisionTreeAlgorithm();
		dta.train(System.getProperty("user.dir") + "/src/main/resources/data_sets_1/training_set.csv", true);
		dta.generateDecisionTree(true);
		dta.printDecisionTree();
		
		List<String> validationDataLines = Utility.loadFile(System.getProperty("user.dir") + "/src/main/resources/data_sets_1/validation_set.csv");
		Instances validationInstances = Utility.loadInstancesFromData(validationDataLines);
		BigDecimal accuracy = dta.validateInstances(validationInstances);
		
		System.out.println(accuracy);
		
	}
	
}
