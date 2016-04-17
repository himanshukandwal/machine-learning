package dev.research.himanshu.ml.playground.textclassification.classifiers;

import java.io.File;
import java.util.Map;

import dev.research.himanshu.ml.playground.decisiontree.model.MLException;

/**
 * enumeration class having the valid classifier enums.
 * 
 * @author Himanshu Kandwal
 *
 */
public enum Classifier {
	
	LR ("\'Logistic Regression\'", new LogisticRegression()),
	NB ("\'Naive Bayes\'", new NaiveBayesClassifier()),
	PC ("\'Perceptron\'", new PerceptronClassifier());
	
	private String classifierName;
	private Classifiable classifiable;
	private File baseDirectory;
	private Map<String, Double> learningParameters;
	
	private Classifier(String classifierName, Classifiable classifiable) {
		this.classifierName = classifierName;
		this.classifiable = classifiable;
	}
	
	public Map<String, Double> getLearningParameters() {
		return learningParameters;
	}
	
	public void setLearningParameters(Map<String, Double> learningParameters) throws MLException {
		this.learningParameters = learningParameters;
		getClassifiable().setLearningParameters(learningParameters);
	}
	
	public String getClassifierName() {
		return classifierName;
	}
	
	public Classifiable getClassifiable() throws MLException {
		classifiable.setBaseDirectory(baseDirectory);
		return classifiable;
	}
	
	public File getBaseDirectory() {
		return baseDirectory;
	}
	
	public void setBaseDirectory(File baseDirectory) {
		this.baseDirectory = baseDirectory;
	}
	
}