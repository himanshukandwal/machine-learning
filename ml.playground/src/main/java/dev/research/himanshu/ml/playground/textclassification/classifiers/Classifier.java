package dev.research.himanshu.ml.playground.textclassification.classifiers;

import java.io.File;

import dev.research.himanshu.ml.playground.decisiontree.model.MLException;

/**
 * enumeration class having the valid classifier enums.
 * 
 * @author Himanshu Kandwal
 *
 */
public enum Classifier {
	
	LR ("\'Logistic Regression\'", null),
	NB ("\'Naive Bayes\'", new NaiveBayesClassifier());
	
	private String classifierName;
	private Classifiable classifiable;
	private File baseDirectory;
	
	private Classifier(String classifierName, Classifiable classifiable) {
		this.classifierName = classifierName;
		this.classifiable = classifiable;
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
