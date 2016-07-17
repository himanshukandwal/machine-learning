package com.utdallas.ml.sensor.anomaly.detection.classifier.impl;

import java.util.LinkedList;
import java.util.List;

import com.utdallas.ml.sensor.anomaly.detection.classifier.impl.UnivariateGaussianClassifier.UnivariateGaussianInstanceClassifier;
import com.utdallas.ml.sensor.anomaly.detection.model.PartitionedDataSet;
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
public class KMeansClassifier extends AbstractClassifiable {

	private List<KMeansInstanceClassifier> instanceClassifiers;
	
	public class KMeansInstanceClassifier {

		private List<Double> clusterCenters; 	// Cluster centers
		private List<Double> maxDistance; 		// Maximum distance to any point in a cluster
		private int numberOfClusters;  			// Number of clusters
		
		public int getNumberOfClusters() {
			return numberOfClusters;
		}

		public void setNumberOfClusters(int numberOfClusters) {
			this.numberOfClusters = numberOfClusters;
		}

		public List<Double> getClusterCenters (int size) {
			if (clusterCenters == null) {
				clusterCenters = new LinkedList<Double>();
				expand(clusterCenters, size);
			}
			
			return clusterCenters;
		}
		
		public List<Double> getClusterCenters() {
			return clusterCenters;
		}

		public void setClusterCenters(List<Double> clusterCenters) {
			this.clusterCenters = clusterCenters;
		}

		public List<Double> getMaxDistance() {
			return maxDistance;
		}

		public void setMaxDistance(List<Double> maxDistance) {
			this.maxDistance = maxDistance;
		}

		public void reset() {
			getClusterCenters().clear();
			getMaxDistance().clear();
			setNumberOfClusters(0);
		}

		public void compute(List<Double> columnData) {
			if (getNumberOfClusters() == 0) {
				System.out.println("Cannot train with 0 clusters.");
				return;
			}

			getClusterCenters(getNumberOfClusters());

			List<Integer> clusterAssignment = new LinkedList<Integer>();
			expand(clusterAssignment, columnData.size());

			List<Integer> prevClusterAssignment = new LinkedList<Integer>();
			expand(prevClusterAssignment, columnData.size());

			double maxDist = 0.0;
			double curDist = 0.0;
			int closestCenter = 0;

			// Seed initial cluster centers by picking k random values from the supplied set. Do not allow repeats.
			double centerValue = 0.0;
			for (int index = 0; index < getClusterCenters().size(); index++) {
				do
					centerValue = columnData.get(PartitionedDataSet.findRandomValue(columnData.size()));
				while (getClusterCenters().contains(centerValue));
				getClusterCenters().set(index, centerValue);
			}
			
			 // Kmeans
			do {
				// Record previous centers for testing convergence later
			    prevClusterAssignment = clusterAssignment;

			    // For each data point find its closest cluster center, update cluster assignment
			    for (int index = 0; index < columnData.size(); index++) {
			      maxDist = Double.MAX_VALUE;
			      
			      for (int innerIndex = 0; innerIndex < getClusterCenters().size(); innerIndex++) {
			        curDist = makeFinite (distance (columnData.get(index), getClusterCenters().get(innerIndex)));
			        
			        if ( curDist < maxDist ) {
			          maxDist = curDist;
			          closestCenter = innerIndex;
			        }
			      }
			      clusterAssignment.set(index, closestCenter);
			    }

			    // Update cluster centers as running averages of assigned points
				List<Integer> timesAssignedToCluster = new LinkedList<Integer>();
				expand(timesAssignedToCluster, getNumberOfClusters());
				
			    for (int index = 0; index < columnData.size(); index++) {
			      double a = ++ timesAssignedToCluster[clusterAssignment[i]];
			      // center = center * proportion + data * proportion
			      m_c[clusterAssignment[i]] = fun::makeFinite( (m_c[clusterAssignment[i]] * ((a - 1.0) / a ))
			                                                 + (data[i] / a) );
			    }
			  } while ( !converged( clusterAssignment, prevClusterAssignment ) );

		}
		
		private void expand(List list, int size) {
			for (int index = 0; index < size; index ++)
				list.add(0);
		}
	}
	
	public List<KMeansInstanceClassifier> getInstanceClassifiers(int size) {
		if (instanceClassifiers == null) {
			instanceClassifiers = new LinkedList<KMeansInstanceClassifier>();
			
			for (int index = 0; index < size; index++)
				instanceClassifiers.add(new KMeansInstanceClassifier());
		}
		
		return instanceClassifiers;
	}

	public List<KMeansInstanceClassifier> getInstanceClassifiers() {
		if (instanceClassifiers == null)
			instanceClassifiers = new LinkedList<KMeansInstanceClassifier>();
			
		return instanceClassifiers;
	}

	public double sample (SensorNode sensorNode) {
		return 0;
	}
	
	@Override
	protected double crossValidate(SensorNodes trainingSensorNodes, SensorNodes trainingAnomalousSensorNodes, double epsilon) {
		return 0;
	}

	@Override
	protected double analyzeData(SensorNode sensorNode) {
		return sample(sensorNode);
	}

	@Override
	protected void populate(SensorNodes trainingSensorNodes) {
		// Grab a column of data and populate a kmeans on it
		if (trainingSensorNodes.isEmpty())
			return;
		
		int dimensions = trainingSensorNodes.getSensorNodes().get(0).getDimension();
		
		for (int dimension = 0; dimension < dimensions; dimension ++) {
			KMeansInstanceClassifier instanceClassifier = new KMeansInstanceClassifier();
			
			List<Double> columnData = new LinkedList<Double>();
			for (SensorNode sensorNode : trainingSensorNodes.getSensorNodes())
				columnData.add(sensorNode.getSensorFeatureData().get(dimension));
			
			instanceClassifier.setNumberOfClusters(1);
			instanceClassifier.compute(columnData);
			getInstanceClassifiers().add(instanceClassifier);
		}
		
		
      for ( unsigned int j = 0; j < okTrainSet[0].size(); j++ ) {
        KMeans k;
        k.setNumClusters(1);

        // Always do the OK data
        col.clear();
        for ( unsigned int i = 0; i < okTrainSet.size(); i++ ) {
          col.push_back( okTrainSet[i][j] );
        }
        k.populate( col );
        m_k[0].push_back( k );

        // Also do the anomalous data if there is any
        if ( m_k.size() == 2 ) {
          col.clear();
          for ( unsigned int i = 0; i < anomTrainSet.size(); i++ ) {
            col.push_back( anomTrainSet[i][j] );
          }
          k.populate( col );
          m_k[1].push_back( k );
        }
      }
	      
		for (int index = 0; index < getNumberOfClusters(); index++)
			getClusterCenters().add(0d);
			
		List<Integer> clusterAssignment, prevClusterAssignment; 
	
		double maxDist = 0.0;
		double curDist = 0.0;
		int closestCenter = 0;
	
		// Seed initial cluster centers by picking k random values from the supplied set. Do not allow repeats.
		double centerValue = 0.0;
		for (int index = 0; index < getNumberOfClusters(); index++) {
			do {
		      centerValue = data[rng.rand(data.size())];
		    } while ( exists( centerValue, m_c ) );
		    m_c[i] = centerValue;
		  } 
	
		  // Kmeans
		  do {
		    // Record previous centers for testing convergence later
		    prevClusterAssignment = clusterAssignment;
	
		    // For each data point find its closest cluster center, update cluster assignment
		    for ( unsigned int i = 0; i < data.size(); i++ ) {
		      maxDist = DBL_MAX;
		      
		      for ( unsigned int j = 0; j < m_c.size(); j++ ) {
		        curDist = fun::makeFinite( fun::dist( data[i], m_c[j] ) );
		        if ( curDist < maxDist ) {
		          maxDist = curDist;
		          closestCenter = j;
		        }
		      }
	
		      clusterAssignment[i] = closestCenter;
		    }
	
		    // Update cluster centers as running averages of assigned points
		    deque<unsigned int> timesAssignedToCluster( m_k, 0 );
		    
		    for ( unsigned int i = 0; i < data.size(); i++ ) {
		      double a = ++timesAssignedToCluster[clusterAssignment[i]];
		      // center = center * proportion + data * proportion
		      m_c[clusterAssignment[i]] = fun::makeFinite( (m_c[clusterAssignment[i]] * ((a - 1.0) / a ))
		                                                 + (data[i] / a) );
		    }
		  } while ( !converged( clusterAssignment, prevClusterAssignment ) );
		  
		  // Calculate max distances
		  m_max.resize( m_k, 0.0 );
		  double dist = 0.0;
	
		  // Compute max distance to assigned cluster's center
		  for ( unsigned int i = 0; i < data.size(); i++ ) {
		    dist = fun::makeFinite( fun::dist(data[i], m_c[clusterAssignment[i]]) );
	
		    if ( dist > m_max[clusterAssignment[i]] ) {
		      m_max[clusterAssignment[i]] = dist;
		    }
		  }
	}

	@Override
	protected void printSpecific() {
		
	}

}
