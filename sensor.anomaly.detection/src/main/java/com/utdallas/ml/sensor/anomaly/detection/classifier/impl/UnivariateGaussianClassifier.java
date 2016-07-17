package com.utdallas.ml.sensor.anomaly.detection.classifier.impl;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import com.utdallas.ml.sensor.anomaly.detection.model.SensorNode;
import com.utdallas.ml.sensor.anomaly.detection.model.SensorNodes;
import com.utdallas.ml.sensor.anomaly.detection.utils.Constants;

/**
 * Univariate classifier for sensor data classification.
 * 
 * @author (Anirudh KV and Himanshu Kandwal)
 *
 */
public class UnivariateGaussianClassifier extends AbstractClassifiable {

	private List<UnivariateGaussianInstanceClassifier> instanceClassifiers;
	
	public class UnivariateGaussianInstanceClassifier {
		
		private Double mean;
		private Double variance;
		
		public void compute(List<Double> columnData) {
			double computedMean, computedVariance;
			computedMean = computedVariance = 0;
			
			for (double featureData : columnData)
				computedMean += featureData;
			computedMean /= columnData.size();
			
			double diff = 0.0;
			for (double featureData : columnData) {
				diff = featureData - computedMean;
				computedVariance += diff * diff;
			}
			computedVariance /= columnData.size();
			
			if (mean == null || variance == null) {
				mean = computedMean;
				variance = computedVariance;
			} else {
				mean = (mean + computedMean) / 2;
				variance = (variance + computedVariance) / 2;
			}
		}
		
		public double getMean() {
			return mean;
		}
		
		public double getVariance() {
			return variance;
		}
		
		public double sample(double value) {
			double diff = (value - mean);
			double numer = Math.exp (-diff * diff / (2.0 * variance));
			double denom = Math.sqrt (variance) * Math.sqrt (2 * Constants.PI);			
			double result = numer / denom;

			return ((result < Math.pow(10, -300)) ? Math.pow(10, -300) : result);
		}
		
		@Override
		public String toString() {
			return " [" + mean + " , " + variance + "] ";
		}
	}
	
	public List<UnivariateGaussianInstanceClassifier> getInstanceClassifiers(int size) {
		if (instanceClassifiers == null) {
			instanceClassifiers = new LinkedList<UnivariateGaussianInstanceClassifier>();
			
			for (int index = 0; index < size; index++)
				instanceClassifiers.add(new UnivariateGaussianInstanceClassifier());
		}
		
		return instanceClassifiers;
	}

	public List<UnivariateGaussianInstanceClassifier> getInstanceClassifiers() {
		return instanceClassifiers;
	}
	
	@Override
	protected void populate(SensorNodes trainingSensorNodes) {
		if (trainingSensorNodes.size() > 0) {
			int dimensions = trainingSensorNodes.getSensorNodes().get(0).getDimension();
			
			for (int dimension = 0; dimension < dimensions; dimension ++) {
				List<Double> dimensionColumnData = new ArrayList<Double>();
				for (SensorNode sensorNode : trainingSensorNodes.getSensorNodes())
					dimensionColumnData.add(sensorNode.getSensorFeatureData().get(dimension));
				
				getInstanceClassifiers(dimensions).get(dimension).compute(dimensionColumnData);
			}
		}	
	}

	@Override
	protected double analyzeData(SensorNode sensorNode) {
		double p = 1.0;
		
	    for (int index = 0; index < getInstanceClassifiers().size(); index ++)
	      p *= getInstanceClassifiers().get(index).sample(sensorNode.getSensorFeatureData().get(index));
	    
	    return p;
	}

	@Override
	protected double crossValidate(SensorNodes trainingSensorNodes, SensorNodes trainingAnomalousSensorNodes, double epsilon) {
		int okCorrectClassifies = 0;
		int anomCorrectClassifies = 0;
		  
		// loop over correct data-set, count number of mis-classifications.
		for (SensorNode sensorNode : trainingSensorNodes.getSensorNodes())	
			okCorrectClassifies += (analyzeData(sensorNode) > epsilon ? 1 : 0);
		
		// loop over anomalous data-set, count number of mis-classifications.
		for (SensorNode sensorNode : trainingAnomalousSensorNodes.getSensorNodes())	
			anomCorrectClassifies += (analyzeData(sensorNode) > epsilon ? 1 : 0);
		
		double np = trainingSensorNodes.size() + trainingAnomalousSensorNodes.size(); // number of points classified
		
		return ( np == 0 ? 1.0 : (double) (okCorrectClassifies + anomCorrectClassifies) / np);
	}

	@Override
	protected void printSpecific() {
		for (UnivariateGaussianInstanceClassifier instanceClassifier : getInstanceClassifiers())
			System.out.println("\t mean : " + instanceClassifier.getMean() + ", variance : " + instanceClassifier.getVariance());
	}
	
}
