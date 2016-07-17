package com.utdallas.ml.sensor.anomaly.detection.model;

import java.util.ArrayList;
import java.util.List;

/**
 * a simple bean model class for sensor nodes. 
 * 
 * @author (Anirudh KV and Himanshu Kandwal)
 *
 */
public class SensorNodes {
	
	private List<SensorNode> sensorNodes;
	
	public List<SensorNode> getSensorNodes() {
		if (sensorNodes == null) 
			sensorNodes = new ArrayList<SensorNode>();
		
		return sensorNodes;
	}
	
	public boolean isEmpty() {
		return getSensorNodes().isEmpty();
	}
	
	public int size() {
		return getSensorNodes().size();
	}
	
	public void addSensorNode(SensorNode sensorNode) {
		getSensorNodes().add(sensorNode);
	}

	@Override
	public String toString() {
		return " nodes : " + getSensorNodes().size();
	}
}
