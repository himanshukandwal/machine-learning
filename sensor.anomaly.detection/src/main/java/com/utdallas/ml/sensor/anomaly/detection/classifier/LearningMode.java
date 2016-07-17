package com.utdallas.ml.sensor.anomaly.detection.classifier;

public enum LearningMode {

	SU ("\'Supervised Learning\'"), 
	SS ("\'Semi-Supervised Learning\'"),
	US ("\'Unsupervised Learning\'");
	
	private String name;
	
	private LearningMode(String name) {
		this.name = name;
	}
	
	public String getName() {
		return name;
	}

	public boolean isUnsupervised() {
		return (this != SU && this != SS);
	}
	
}
