package com.utdallas.ml.sensor.anomaly.detection.utils;

import java.util.LinkedList;
import java.util.List;

import com.utdallas.ml.sensor.anomaly.detection.model.SensorNodes;

/**
 * class handling all the matrix specific tasks.
 * 
 * @author (Anirudh KV and Himanshu Kandwal)
 *
 */
public class MatrixUtils {
	
	public static double covariance (int row, int column, SensorNodes trainingData, List<Double> means) {
		if (!trainingData.isEmpty()) {
			double sum = 0.0;
			
			for (int rowIndex = 0; rowIndex < trainingData.size(); rowIndex ++)
				sum += (trainingData.getSensorNodes().get(rowIndex).getSensorFeatureData().get(row) - means.get(row)) 
						* (trainingData.getSensorNodes().get(rowIndex).getSensorFeatureData().get(column) - means.get(column));
			
			return sum / trainingData.size();
		}
		return 0;
	}
	
	public static double det(List<List<Double>> matrix) {
		return laplaceDet(matrix);
	}
	
	public static double laplaceDet(List<List<Double>> matrix) {
		if (matrix.size() == 3 ) {
		    return (matrix.get(0).get(0) * (matrix.get(1).get(1) * matrix.get(2).get(2) - matrix.get(1).get(2) * matrix.get(2).get(1))
		    		+ matrix.get(0).get(1) * (matrix.get(1).get(2) * matrix.get(2).get(0) - matrix.get(2).get(2) * matrix.get(1).get(0))
		    		+ matrix.get(0).get(2) * (matrix.get(1).get(0) * matrix.get(2).get(1) - matrix.get(1).get(1) * matrix.get(2).get(0)));
		}
		else if (matrix.size() == 2) {
			return matrix.get(0).get(0) * matrix.get(1).get(1) - matrix.get(0).get(1) * matrix.get(1).get(0);
		}
		else if (matrix.size() == 1) {
			return matrix.get(0).get(0);
		}
		else if (matrix.size() == 0 ) {
		    return 0.0;
		}
		else {
			double sum = 0.0;
		    List<List<Double>> minor;
		    
		    for (int index = 0; index < matrix.size(); index ++) {
		      minor = getMinorMatrix(matrix, 0, index );

		      sum += ((index % 2 == 1) ? -1.0 : 1.0) * matrix.get(0).get(index) * laplaceDet (minor);
		    }
		    return sum;
		}
	}
	
	public static List<List<Double>> inverse(List<List<Double>> matrix) {
		if (matrix.size() == 3) {
			double A, B, C, D, E, F, G, H, K, a, b, c, d, e, f, g, h, k;
			
		    a = matrix.get(0).get(0); b = matrix.get(0).get(1); c = matrix.get(0).get(2);
		    d = matrix.get(1).get(0); e = matrix.get(1).get(1); f = matrix.get(1).get(2);
		    g = matrix.get(2).get(0); h = matrix.get(2).get(1); k = matrix.get(2).get(2);

		    A = e*k - f*h; D = c*h - b*k; G = b*f - c*e;
		    B = f*g - d*k; E = a*k - c*g; H = c*d - a*f;
		    C = d*h - e*g; F = g*b - a*h; K = a*e - b*d;

		    List<List<Double>> result = new LinkedList<List<Double>>();
		    for (int index = 0; index < matrix.size(); index ++) {
		    	List<Double> innerList = new LinkedList<Double>();
		    	for (int innerIndex = 0; innerIndex < matrix.size(); innerIndex ++)
		    		innerList.add(0d);
		    	result.add(innerList);	
			}
		    
		    result.get(0).set(0, A);	result.get(0).set(1, D);	result.get(0).set(2, G);
		    result.get(1).set(0, B);	result.get(1).set(1, E);	result.get(1).set(2, H);
		    result.get(2).set(0, C);	result.get(2).set(1, F);	result.get(2).set(2, K);
		    
			return multiply (result, 1.0 / det(matrix));
		}
		else if (matrix.size() == 2) {
			List<List<Double>> result = new LinkedList<List<Double>>();
		    for (int index = 0; index < matrix.size(); index ++) {
		    	List<Double> innerList = new LinkedList<Double>();
		    	for (int innerIndex = 0; innerIndex < matrix.size(); innerIndex ++)
		    		innerList.add(0d);
		    	result.add(innerList);	
			}

		    result.get(0).set(0, matrix.get(1).get(1));		result.get(0).set(1, -matrix.get(0).get(1));
		    result.get(1).set(0, -matrix.get(1).get(0));	result.get(1).set(1, matrix.get(0).get(0));
		    
		    return multiply (result, 1.0 / det(matrix));
		}
		else if (matrix.size() == 1) {
			List<List<Double>> result = new LinkedList<List<Double>>();
			for (int index = 0; index < matrix.size(); index ++) {
		    	List<Double> innerList = new LinkedList<Double>();
		    	for (int innerIndex = 0; innerIndex < matrix.size(); innerIndex ++)
		    		innerList.add(0d);
		    	result.add(innerList);	
			}
			
			result.get(0).set(0, 1/matrix.get(0).get(0));
			return result;
		}
		else if (matrix.size() == 0) {
			return matrix;
		}
		else {
			// Assumes square matrix.
			
			List<List<Double>> result = new LinkedList<List<Double>>();
			for (int index = 0; index < matrix.size(); index ++) {
		    	List<Double> innerList = new LinkedList<Double>();
		    	for (int innerIndex = 0; innerIndex < matrix.size(); innerIndex ++)
		    		innerList.add(0d);
		    	result.add(innerList);	
			}
			
			double scalar = 1.0 / det(matrix);

			for (int index = 0; index < matrix.size(); index ++) {
		    	for (int innerIndex = 0; innerIndex < matrix.size(); innerIndex ++) {
		    		result.get(index).set(innerIndex, scalar * det(getMinorMatrix(matrix, index, innerIndex)));
		    		if (((index + innerIndex) % 2) == 1)
		    			result.get(index).set(innerIndex , -result.get(index).get(innerIndex));
		    	}	
			}
			
			return result;
		}
	}
	
	public static List<List<Double>> getMinorMatrix(List<List<Double>> matrix, int row, int column) {
		if (matrix.isEmpty()) {
			return matrix;
		}
		
		List<List<Double>> resultMatrix = new LinkedList<List<Double>>();
		for (int index = 0; index < matrix.size(); index++) {
			if (index != row) {
				List<Double> rowEntry = new LinkedList<Double>();
				
				for (int innerIndex = 0; innerIndex < resultMatrix.get(0).size(); innerIndex++) {
					if (innerIndex != column)
						rowEntry.add(resultMatrix.get(index).get(innerIndex));
				}
			}
		}
		return resultMatrix;
	}

	public static double dotProduct(List<Double> u, List<Double> v) {
		double sum = 0.0;
		for (int index = 0; index < u.size(); index++) {
		    sum += (u.get(index) * v.get(index));
		}
		
		return sum;
	}

	public static List<Double> minus(List<Double> u, List<Double> v) {
		List<Double> result = new LinkedList<Double>();
		for (int index = 0; index < u.size(); index ++)
		    result.add (u.get(index) - v.get(index));
		 
		return result;
	}
	
	public static List<List<Double>> multiply(List<List<Double>> u, Double value) {
		for (int index = 0; index < u.size(); index ++)
			for (int innerIndex = 0; innerIndex < u.get(0).size(); innerIndex ++)
				u.get(index).set(innerIndex, u.get(index).get(innerIndex) * value);
		
		return u;
	}

	public static List<Double> multiply(List<List<Double>> u, List<Double> v) {
		List<Double> result = new LinkedList<Double>();
		for (int index = 0; index < u.size(); index ++)
			result.add(dotProduct(u.get(index), v));
		  
		return result;
	}

}
