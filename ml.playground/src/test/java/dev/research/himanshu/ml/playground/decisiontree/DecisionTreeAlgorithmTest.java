package dev.research.himanshu.ml.playground.decisiontree;

import junit.framework.TestCase;

public class DecisionTreeAlgorithmTest extends TestCase {

	public void testDecisionTree() throws Exception {
		DecisionTreeAlgorithm dta = new DecisionTreeAlgorithm();
		dta.train(
				"/Users/Heman/Documents/workstation/Developement_Studio/Java_Laboratory/Leisure_WorkZones/Eclipse_Workspaces/Machine-Learning/ml.playground/src/main/resources/data_sets_1/test_set.csv");
		
		System.out.println(" -- loaded training data !");
		
		dta.generateDecisionTree();
		dta.printDecisionTree();
	}
}
