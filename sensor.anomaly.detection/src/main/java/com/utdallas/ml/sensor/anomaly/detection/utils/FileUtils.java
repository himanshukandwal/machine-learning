package com.utdallas.ml.sensor.anomaly.detection.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FilenameFilter;

import com.utdallas.ml.sensor.anomaly.detection.classifier.impl.AbstractClassifiable;
import com.utdallas.ml.sensor.anomaly.detection.model.MLException;
import com.utdallas.ml.sensor.anomaly.detection.model.SensorNode;
import com.utdallas.ml.sensor.anomaly.detection.model.SensorNodes;

/**
 * class handling all the file system specific tasks.
 * 
 * @author (Anirudh KV and Himanshu Kandwal)
 *
 */
public class FileUtils {

	public static void locateAndReadData (AbstractClassifiable classifiable, File baseDirectory) throws MLException {
		File[] files = baseDirectory.listFiles(new FilenameFilter() {
			public boolean accept(File dir, String name) {
				File file = new File(dir, name);
				if (file.isFile() && name.endsWith(Constants.EXTENSION))
					return true;
				return false;
			}
		});
		
		readFiles(classifiable, files);
	}
	
	public static void readFile(AbstractClassifiable classifiable, File baseDirectory, String filename) throws MLException {
		readFiles(classifiable, new File[] { new File(baseDirectory, filename) });	
	}

	public static void readFile(AbstractClassifiable classifiable, File file) throws MLException {
		readFiles(classifiable, new File [] { file });	
	}
	
	public static void readFiles(AbstractClassifiable classifiable, File[] files) throws MLException {
		try {
			SensorNodes correctData = new SensorNodes();
			SensorNodes anamolousData = new SensorNodes();
			
			for (File file : files) {
				BufferedReader bufferedReader = null;

				bufferedReader = new BufferedReader(new FileReader(file));
				String line = null;

				while ((line = bufferedReader.readLine()) != null) {
					line = line.trim();
					boolean isAnomalousInstance = false;

					if (!line.isEmpty()) {
						SensorNode sensorNode = new SensorNode();

						for (String word : line.split(Constants.DATA_DELIMITER)) {
							if (word.equals(Constants.NAN) || word.equals(Constants.INF)) {
								isAnomalousInstance = true;
								sensorNode.addFeature(Double.MAX_VALUE);
							} else
								sensorNode.addFeature(Double.valueOf(word));
						}

						if (isAnomalousInstance) {
							if (classifiable.getClassifierType().getLearningMode().isUnsupervised())
								anamolousData.addSensorNode(sensorNode);
						} else {
							correctData.addSensorNode(sensorNode);
						}
					}
				}
				bufferedReader.close();
			}

			classifiable.setCompleteCorrectData(correctData);
			classifiable.setCompleteAnomalousData(anamolousData);
			
		} catch (Exception exception) {
			throw new MLException(" caught in exception while loading text documents !");
		}
		
		System.out.println(" -- read training data !");	
	}
	
}
