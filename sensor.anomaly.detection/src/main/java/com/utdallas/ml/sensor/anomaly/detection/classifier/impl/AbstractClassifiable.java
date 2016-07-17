package com.utdallas.ml.sensor.anomaly.detection.classifier.impl;

import java.util.LinkedList;
import java.util.List;

import com.utdallas.ml.sensor.anomaly.detection.classifier.Classifiable;
import com.utdallas.ml.sensor.anomaly.detection.classifier.ClassifierType;
import com.utdallas.ml.sensor.anomaly.detection.model.MLException;
import com.utdallas.ml.sensor.anomaly.detection.model.PartitionedDataSet;
import com.utdallas.ml.sensor.anomaly.detection.model.Point;
import com.utdallas.ml.sensor.anomaly.detection.model.SensorNode;
import com.utdallas.ml.sensor.anomaly.detection.model.SensorNodes;
import com.utdallas.ml.sensor.anomaly.detection.utils.Constants;
import com.utdallas.ml.sensor.anomaly.detection.utils.FileUtils;
import com.utdallas.ml.sensor.anomaly.detection.utils.SensorDataStatistic;

/**
 * abstract class for handling the background common code.
 * 
 * @author (Anirudh KV and Himanshu Kandwal)
 *
 */
public abstract class AbstractClassifiable implements Classifiable {
	
	private int iterations;
	private double cv;
	private double bestEpsilon;
	private ClassifierType classifierType;
	private List<String> header;
	private SensorNodes completeCorrectData;
	private SensorNodes completeAnomalousData;
	private SensorNodes testingCorrectData;
	private SensorNodes testingAnomalousData;
	
	public int getIterations() {
		return iterations;
	}
	
	public void setIterations(int iterations) {
		this.iterations = iterations;
	}
	
	public List<String> getHeader() {
		if (header == null)
			header = new LinkedList<String>();
		return header;
	}

	public void setHeader(List<String> header) {
		this.header = header;
	}

	public SensorNodes getCompleteCorrectData() {
		return completeCorrectData;
	}
	
	public void setCompleteCorrectData(SensorNodes correctData) {
		this.completeCorrectData = correctData;
	}
	
	public SensorNodes getCompleteAnomalousData() {
		return completeAnomalousData;
	}
	
	public void setCompleteAnomalousData(SensorNodes anomalousData) {
		this.completeAnomalousData = anomalousData;
	}
	
	public ClassifierType getClassifierType() {
		return classifierType;
	}
	
	public void setClassifierType(ClassifierType classifierType) {
		this.classifierType = classifierType;
		setIterations(classifierType.getLearningParameters().get(Constants.REPETITIONS).intValue());
		setCv(classifierType.getLearningParameters().get(Constants.CV));
	}
	
	public double getCv() {
		return cv;
	}
	
	public void setCv(double cv) {
		this.cv = cv;
	}
	
	public double getBestEpsilon() {
		return bestEpsilon;
	}
	
	public void setBestEpsilon(double bestEpsilon) {
		this.bestEpsilon = bestEpsilon;
	}
	
	public SensorNodes getTestingCorrectData() {
		return testingCorrectData;
	}
	
	public void setTestingCorrectData(SensorNodes testingCorrectData) {
		this.testingCorrectData = testingCorrectData;
	}
	
	public SensorNodes getTestingAnomalousData() {
		return testingAnomalousData;
	}
	
	public void setTestingAnomalousData(SensorNodes testingAnomalousData) {
		this.testingAnomalousData = testingAnomalousData;
	}
	
	public void train() throws MLException {
		/* load training data */
		FileUtils.locateAndReadData(this, getClassifierType().getBaseDirectory());
			
		// Keep 70% of samples for training purposes. Don't touch rest until test()!
		PartitionedDataSet partitionedDataCorrectSet = PartitionedDataSet.partition(getCompleteCorrectData(), 0.7d);
		PartitionedDataSet partitionedDataAnomalousDataSet = PartitionedDataSet.partition(getCompleteAnomalousData(), 0.7d);
				
		setTestingCorrectData(partitionedDataCorrectSet.getRemainingData());
		setTestingAnomalousData(partitionedDataAnomalousDataSet.getRemainingData());
		
		double bestAccuracy = 0;
		double bestEpsilon = 0;
		for (int loop = 0; loop < getIterations(); loop++) {
			PartitionedDataSet iterationPartitionedDataCorrectSet = PartitionedDataSet.partition(partitionedDataCorrectSet.getMainData(), getCv());
			PartitionedDataSet iterationPartitionedDataAnomalousSet = new PartitionedDataSet();
			
			if (getClassifierType().getLearningMode().isUnsupervised()) 
				iterationPartitionedDataAnomalousSet = PartitionedDataSet.partition(partitionedDataAnomalousDataSet.getMainData(), getCv());
			
			// populate the classifier.
			populate(iterationPartitionedDataCorrectSet.getMainData());
			
			// compute epsilon.
			double itrEpsilon = calculateEpsilon(iterationPartitionedDataCorrectSet.getMainData(), iterationPartitionedDataAnomalousSet.getMainData());
			
			// Test accuracy on the cross validation sets.
			double accuracy = crossValidate (iterationPartitionedDataCorrectSet.getRemainingData(), iterationPartitionedDataAnomalousSet.getRemainingData(), itrEpsilon);
			
			if (accuracy > bestAccuracy) {
				bestAccuracy = accuracy;
				bestEpsilon = itrEpsilon;
			}
		}
		setBestEpsilon(bestEpsilon);
	}

	public double test() throws MLException {
		return crossValidate(getTestingCorrectData(), getTestingAnomalousData(), getBestEpsilon());
	}
	
	protected abstract double crossValidate(SensorNodes trainingSensorNodes, SensorNodes trainingAnomalousSensorNodes, double epsilon);

	protected abstract double analyzeData(SensorNode sensorNode);
	
	protected abstract void populate(SensorNodes trainingSensorNodes);
	
	protected abstract void printSpecific();
	
	protected double calculateEpsilon(SensorNodes trainingSensorNodes, SensorNodes trainingAnomalousSensorNodes) {
		
		if (trainingSensorNodes.isEmpty() && trainingAnomalousSensorNodes.isEmpty())
			return 0d;

		double value;

		// Get minimum probability from the correctTrainingSensorNodes
		SensorDataStatistic correctDataStatistic = new SensorDataStatistic(Double.MAX_VALUE);
		for (SensorNode sensorNode : trainingSensorNodes.getSensorNodes()) {
			value = analyzeData(sensorNode);

			if (value < correctDataStatistic.getEpsilon())
				correctDataStatistic.setEpsilon(value);

			if (value < correctDataStatistic.getProbability())
				correctDataStatistic.setProbability(value);
		}

		if (correctDataStatistic.getProbability() == Double.MAX_VALUE)
			correctDataStatistic.setProbability(0d);

		// Get maximum probability from the anamolousTrainingSensorNodes
		SensorDataStatistic anamolousSensorDataStatistic = new SensorDataStatistic(Double.MIN_VALUE);
		for (SensorNode sensorNode : trainingAnomalousSensorNodes.getSensorNodes()) {
			value = analyzeData(sensorNode);

			if (value > anamolousSensorDataStatistic.getEpsilon())
				anamolousSensorDataStatistic.setEpsilon(value);

			if (value > anamolousSensorDataStatistic.getProbability())
				anamolousSensorDataStatistic.setProbability(value);
		}

		if (correctDataStatistic.getProbability() == Double.MAX_VALUE)
			correctDataStatistic.setProbability(0d);

		if (trainingSensorNodes.isEmpty())
			return anamolousSensorDataStatistic.getProbability();

		if (trainingAnomalousSensorNodes.isEmpty())
			return correctDataStatistic.getProbability();

		value = correctDataStatistic.getProbability() + anamolousSensorDataStatistic.getProbability();
		double denom = (trainingAnomalousSensorNodes.isEmpty() ? trainingSensorNodes.size() : ((double) trainingSensorNodes.size() / trainingAnomalousSensorNodes.size()));

		return value / denom;
	}
	
	public void print() throws MLException {
		System.out.println("\n === summary ===");
		System.out.println(" > Data size : " + getCompleteCorrectData().size());
		System.out.println(" > Anomalous Data size : " + getCompleteAnomalousData().size());
		System.out.println(" > Algorithm : " + getClassifierType().getClassifierName());
		System.out.println(" > Algorithm Learning mode : " + getClassifierType().getLearningMode().getName());
		System.out.println(" > Epsilon : " + getBestEpsilon());
		
		// print classifier specific information.
		printSpecific();
	}
	
	protected int distance (double point1, double point2) {
		return (int) Math.sqrt(Math.pow(point2 - point1, 2));
	}
	
	protected int distance (Point point1, Point point2) {
		return (int) Math.sqrt(Math.pow(point2.getX1() - point1.getX1(), 2) 
				+ Math.pow(point2.getX2() - point1.getX2(), 2)
				+ Math.pow(point2.getX3() - point1.getX3(), 2));
	}
	
	protected double makeFinite(double value) {
		if (value > Double.MAX_VALUE)
			value = Double.MAX_VALUE;
		else if (value < Double.MIN_VALUE)
			value = Double.MIN_VALUE;
		
		return value;
	}

}
