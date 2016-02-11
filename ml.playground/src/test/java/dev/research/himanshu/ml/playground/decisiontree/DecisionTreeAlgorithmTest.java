package dev.research.himanshu.ml.playground.decisiontree;

import junit.framework.TestCase;

public class DecisionTreeAlgorithmTest extends TestCase {

	public void testDecisionTree() throws Exception {
		DecisionTreeAlgorithm dta = new DecisionTreeAlgorithm();
		dta.train(System.getProperty("user.dir") + "/src/main/resources/data_sets_1/training_set.csv");
		dta.generateDecisionTree();
		dta.printDecisionTree();
	}
	
}
