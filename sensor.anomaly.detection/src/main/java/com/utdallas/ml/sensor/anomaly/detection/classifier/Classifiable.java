package com.utdallas.ml.sensor.anomaly.detection.classifier;

import com.utdallas.ml.sensor.anomaly.detection.model.MLException;

/**
 * generic interface presenting a simple classifiable interface. 
 * 
 * @author (Anirudh KV and Himanshu Kandwal)
 *
 */
public interface Classifiable {
	
	public void setClassifierType (ClassifierType classifierType);

	public void train() throws MLException;
	
	public double test() throws MLException;
	
	public void print() throws MLException;

}
