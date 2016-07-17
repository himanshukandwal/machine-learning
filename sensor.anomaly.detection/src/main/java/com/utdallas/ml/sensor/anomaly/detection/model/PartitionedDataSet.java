package com.utdallas.ml.sensor.anomaly.detection.model;

import java.util.Random;

/**
 * A simple sensor data partitioner.
 * 
 * @author (Anirudh KV and Himanshu Kandwal)
 *
 */
public class PartitionedDataSet {
	
	private SensorNodes mainData;
	private SensorNodes remainingData;
	private static Random random = new Random();
	
	public PartitionedDataSet() {
		mainData = new SensorNodes();
		remainingData = new SensorNodes();
	}
	
	public SensorNodes getMainData() {
		return mainData;
	}
	
	public SensorNodes getRemainingData() {
		return remainingData;
	}
	
	public static PartitionedDataSet partition(SensorNodes sensorNodes, double pct) {
		PartitionedDataSet partitionedDataSet = new PartitionedDataSet();
		
		if (!sensorNodes.isEmpty()) {
			int size = (int) (sensorNodes.size() * pct);
			int start = findRandomValue (sensorNodes.size() - size);
			
			for (int index = 0; index < sensorNodes.size(); index++) {
				if (index > start && index <= (start + size)) 
					partitionedDataSet.mainData.addSensorNode(sensorNodes.getSensorNodes().get(index));
				else 
					partitionedDataSet.remainingData.addSensorNode(sensorNodes.getSensorNodes().get(index));
			}
		}
		
		return partitionedDataSet;
	}

	public static int findRandomValue(int high) {
		return findRandomValue(0, high);
	}
	
	public static int findRandomValue(int low, int high) {
		return low + (random.nextInt(high-low));
	}

}
