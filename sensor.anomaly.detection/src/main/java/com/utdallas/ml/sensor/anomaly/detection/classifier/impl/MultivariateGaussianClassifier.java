package com.utdallas.ml.sensor.anomaly.detection.classifier.impl;

import java.util.LinkedList;
import java.util.List;

import com.utdallas.ml.sensor.anomaly.detection.model.SensorNode;
import com.utdallas.ml.sensor.anomaly.detection.model.SensorNodes;
import com.utdallas.ml.sensor.anomaly.detection.utils.Constants;
import com.utdallas.ml.sensor.anomaly.detection.utils.MatrixUtils;

/**
 * Multivariate classifier for sensor data classification.
 * 
 * @author (Anirudh KV and Himanshu Kandwal)
 *
 */
public class MultivariateGaussianClassifier extends AbstractClassifiable {

	private List<Double> means;
	private List<List<Double>> covarianceMatrix;
	private List<List<Double>> sigmaInverseMatrix;
	private double sigmaDeterminant;
	
	public List<Double> getMeans() {
		if (means == null)
			means = new LinkedList<Double>();
		
		return means;
	}

	public void setMeans(List<Double> means) {
		this.means = means;
	}

	public List<List<Double>> getCovarianceMatrix() {
		if (covarianceMatrix == null)
			covarianceMatrix = new LinkedList<List<Double>>();
		
		return covarianceMatrix;
	}

	public void setCovarianceMatrix(List<List<Double>> covarianceMatrix) {
		this.covarianceMatrix = covarianceMatrix;
	}

	public List<List<Double>> getSigmaInverseMatrix() {
		if (sigmaInverseMatrix == null)
			sigmaInverseMatrix = new LinkedList<List<Double>>();
		
		return sigmaInverseMatrix;
	}

	public void setSigmaInverseMatrix(List<List<Double>> sigmaInverseMatrix) {
		this.sigmaInverseMatrix = sigmaInverseMatrix;
	}

	public double getSigmaDeterminant() {
		return sigmaDeterminant;
	}

	public void setSigmaDeterminant(double sigmaDeterminant) {
		this.sigmaDeterminant = sigmaDeterminant;
	}
	
	public void reset() {
		getMeans().clear();
		getCovarianceMatrix().clear();
		getSigmaInverseMatrix().clear();
		setSigmaDeterminant(0d);
	}

	public double sample (SensorNode sensorNode) {
	  List<Double> diff = MatrixUtils.minus (sensorNode.getSensorFeatureData(), getMeans());

	  double numer = Math.exp (-0.5d * MatrixUtils.dotProduct(diff, MatrixUtils.multiply(getSigmaInverseMatrix(), diff)));
	  double denom = Math.pow ((2 * Constants.PI), getMeans().size() / 2.0 ) * Math.sqrt(getSigmaDeterminant());

	  if (denom == 0.0)
	    return 0.0; 
	  
	  double result = numer / denom;

	  if (result < Math.pow (10, -300))
	    result = Math.pow (10, -300);
	  
	  else if (result > Math.pow (10, 300))
	    result = Math.pow (10, 300);
	  
	  return result;
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
	protected double analyzeData(SensorNode sensorNode) {
		return sample(sensorNode);
	}

	@Override
	protected void populate(SensorNodes trainingSensorNodes) {
		reset();
		
		if (trainingSensorNodes.size() > 0) {
			// calculate means for each column.
			double sum = 0.0;
			int dimensions = trainingSensorNodes.getSensorNodes().get(0).getDimension();
			
			for (int dimension = 0; dimension < dimensions; dimension ++ ) {
				sum = 0.0;

			    for (SensorNode sensorNode : trainingSensorNodes.getSensorNodes())
			      sum += sensorNode.getSensorFeatureData().get(dimension);
	    
			    getMeans().add(sum / trainingSensorNodes.size());
			}
			
			// compute covariance matrix.
			for (int index = 0; index < dimensions; index ++) {
				List<Double> innerList = new LinkedList<Double>();
				
				for (int innerIndex = 0; innerIndex < dimensions; innerIndex++)
					innerList.add(MatrixUtils.covariance(index, innerIndex, trainingSensorNodes, getMeans()));
					
				getCovarianceMatrix().add(innerList);
			}
				
			setSigmaInverseMatrix(MatrixUtils.inverse(getCovarianceMatrix()));
			setSigmaDeterminant(MatrixUtils.det(getCovarianceMatrix()));
		}
	}

	@Override
	protected void printSpecific() {
		
	}

}
