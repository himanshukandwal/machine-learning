package com.utdallas.ml.sensor.anomaly.detection;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import com.utdallas.ml.sensor.anomaly.detection.classifier.Classifiable;
import com.utdallas.ml.sensor.anomaly.detection.classifier.ClassifierType;
import com.utdallas.ml.sensor.anomaly.detection.classifier.LearningMode;
import com.utdallas.ml.sensor.anomaly.detection.model.MLException;
import com.utdallas.ml.sensor.anomaly.detection.utils.Constants;

/**
 * Sensor data anomaly classifier.
 * 
 * @author (Anirudh KV and Himanshu Kandwal)
 *
 */
public class AnamolyDetectionClassifier {
	
	private static Map<String, Double> parseArguments(ClassifierType classifier, String[] args) {
		Map<String, Double> parametermap = new HashMap<String, Double>();
		
		parametermap.put(Constants.CV, 0.85d);
		for (int index = 0; index < args.length; index ++) {
			if (index == 3)
				parametermap.put(Constants.REPETITIONS, Double.valueOf(args[index]));
		}
		
		return parametermap;
	}
	
	public static void main(String[] args) throws MLException {

		if (args.length < 4) {
			System.out.println(" please provide valid inputs. kindly provide input in the following format :");
			System.out.println();
			System.out.println(" \t AnamolyDetectionClassifier <directory-location-containing-test-and-training-files> <classifier-name> <learning-mode> <repetitions> ");
			System.out.println();
			System.out.println(" -----------------------------------------------------------------------------");
			System.out.println(" classifier-name 	: UG or MG or KM");
			System.out.println(" learning-mode 		: SU or SS or US");
			System.out.println(" repetitions		: 10");
			System.out.println(" -----------------------------------------------------------------------------");
			System.out.println(" example :");
			System.out.println(" \t AnamolyDetectionClassifier ./data UG SU 10");
			System.out.println(" \t AnamolyDetectionClassifier ./data MG SS 30");
			System.out.println(" \t AnamolyDetectionClassifier ./data KM US 50");
			System.exit(1);
		} 
		
		String dataDirectoryLocation = args[0];
		File baseDirectory = new File (dataDirectoryLocation);
		
		if (!(baseDirectory.exists() && baseDirectory.isDirectory())) {
			System.out.println(" please check the data directory location : " + dataDirectoryLocation);
			System.out.println(" either it does not exists, or if it exists, its not a directory (folder).");
			System.exit(1);
		}
		
		ClassifierType classifier = ClassifierType.valueOf(args[1]);
		if (classifier == null) {
			System.out.println(" invalid value for classifier specified. Please pass values only :");
			for (ClassifierType classifierItem : ClassifierType.values())
				System.out.println("  -  " + classifierItem.name());
			
			System.exit(1);
		}
		classifier.setBaseDirectory(baseDirectory);
		
		LearningMode learningMode = LearningMode.valueOf(args[2]);
		if (learningMode == null) {
			System.out.println(" invalid value for learning mode specified. Please pass values only :");
			for (LearningMode mode : LearningMode.values())
				System.out.println("  -  " + mode.name());
			
			System.exit(1);
		}
		classifier.setLearningMode(learningMode);
		
		Map<String, Double> parameterMap = parseArguments(classifier, args);
		
		try {
			if (parameterMap != null)
				classifier.setLearningParameters(parameterMap);
			
			System.out.println(" -- running " + classifier.getClassifierName() + " in learning mode : " + classifier.getLearningMode().getName());
			Classifiable classifiable = classifier.getClassifiable();
			
			classifiable.train();
			
			double accuracy = classifiable.test();
			
			classifiable.print();
			
			System.out.println(" -----------------------------");
			System.out.println(" total accuracy : " + (accuracy * 100));
			
		} catch (MLException e) {
			throw new MLException(" exception while performing text classification. ", e);
		}
	}

}
