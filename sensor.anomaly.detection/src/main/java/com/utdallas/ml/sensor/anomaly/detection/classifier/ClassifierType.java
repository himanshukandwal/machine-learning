package com.utdallas.ml.sensor.anomaly.detection.classifier;

import java.io.File;
import java.util.Map;

import com.utdallas.ml.sensor.anomaly.detection.classifier.impl.MultivariateGaussianClassifier;
import com.utdallas.ml.sensor.anomaly.detection.classifier.impl.UnivariateGaussianClassifier;
import com.utdallas.ml.sensor.anomaly.detection.model.MLException;

/**
 * enumeration class having the valid classifier enums.
 * 
 * @author (Anirudh KV and Himanshu Kandwal)
 *
 */
public enum ClassifierType {
	
	UG ("\'Univariate Gaussion Algorithm \'", new UnivariateGaussianClassifier()),
	MG ("\'Multivariate Gaussion Algorithm\'", new MultivariateGaussianClassifier()),
	KM ("\'Kmeans Algorithm\'", null);
	
	private String classifierName;
	private Classifiable classifiable;
	private File baseDirectory;
	private LearningMode learningMode;
	private Map<String, Double> learningParameters;
	
	private ClassifierType (String classifierName, Classifiable classifiable) {
		this.classifierName = classifierName;
		this.classifiable = classifiable;
	}
	
	public Map<String, Double> getLearningParameters() {
		return learningParameters;
	}
	
	public void setLearningParameters(Map<String, Double> learningParameters) throws MLException {
		this.learningParameters = learningParameters;
	}
	
	public String getClassifierName() {
		return classifierName;
	}
	
	public Classifiable getClassifiable() throws MLException {
		classifiable.setClassifierType(this);
		return classifiable;
	}
	
	public File getBaseDirectory() {
		return baseDirectory;
	}
	
	public void setBaseDirectory(File baseDirectory) {
		this.baseDirectory = baseDirectory;
	}
	
	public LearningMode getLearningMode() {
		return learningMode;
	}
	
	public void setLearningMode(LearningMode learningMode) {
		this.learningMode = learningMode;
	}
	
}