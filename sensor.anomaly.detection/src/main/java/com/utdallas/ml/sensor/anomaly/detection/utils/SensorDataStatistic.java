package com.utdallas.ml.sensor.anomaly.detection.utils;

/**
 * class for storing min and max probability values.
 * 
 * @author (Anirudh KV and Himanshu Kandwal)
 *
 */
public class SensorDataStatistic {

	private double probability;
	private double epsilon;
	
	public SensorDataStatistic(Double value) {
		this.probability = this.epsilon = value;
	}
	
	public double getProbability() {
		return probability;
	}
	
	public void setProbability(double minProbability) {
		this.probability = minProbability;
	}
	
	public double getEpsilon() {
		return epsilon;
	}
	
	public void setEpsilon(double epsilon) {
		this.epsilon = epsilon;
	}
	
}
