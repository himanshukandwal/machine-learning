package com.utdallas.ml.sensor.anomaly.detection.model;

import java.util.LinkedList;
import java.util.List;

/**
 * a simple bean model class for sensor node. 
 * 
 * @author (Anirudh KV and Himanshu Kandwal)
 *
 */
public class SensorNode {
	
	private List<Double> sensorFeatureData;
	
	public List<Double> getSensorFeatureData() {
		if (sensorFeatureData == null) 
			sensorFeatureData = new LinkedList<Double>();
		
		return sensorFeatureData;
	}
	
	public void addFeature(Double value) {
		getSensorFeatureData().add(value);
	}

	public int getDimension() {
		return sensorFeatureData.size();
	}
	
	@Override
	public String toString() {
		return getSensorFeatureData().size() + "";
	}
	
}
