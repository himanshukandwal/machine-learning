package dev.research.himanshu.ml.playground.textclassification;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import dev.research.himanshu.ml.playground.decisiontree.model.MLException;
import dev.research.himanshu.ml.playground.textclassification.classifiers.Classifier;

/**
 * class handling the core work-flow for text classification.
 * 
 * @author Himanshu Kandwal
 *
 */
public class TextClassifier {

	private File baseDirectory;
	private Classifier classifier;
	private boolean	stopWordsUsed;
	
	public TextClassifier(File baseLocation, Classifier classifier) {
		this.baseDirectory = baseLocation;
		this.classifier = classifier;
	}

	public File getBaseDirectory() {
		return baseDirectory;
	}
	
	public void setBaseDirectory(File baseDirectory) {
		this.baseDirectory = baseDirectory;
	}
	
	public Classifier getClassifier() {
		return classifier;
	}
	
	public void setClassifier(Classifier classifier) {
		this.classifier = classifier;
	}
	
	public boolean isStopWordsUsed() {
		return stopWordsUsed;
	}
	
	public void setStopWordsUsed(boolean stopWordsUsed) {
		this.stopWordsUsed = stopWordsUsed;
	}
	
	public void train() throws MLException {
		try {
			/** perform training **/
			getClassifier().getClassifiable().train(isStopWordsUsed());
		} catch (MLException e) {
			throw new MLException(" exception during training !", e);
		}
	}
	
	public double test() throws MLException {
		try {
			/** perform testing **/
			return getClassifier().getClassifiable().test();
		} catch (MLException e) {
			throw new MLException(" exception during training !", e);
		}
	}
	
	public void reset() throws MLException {
		getClassifier().getClassifiable().reset();
	}
	
	private static Map<String, Double> parseArguments(String[] args) {
		Map<String, Double> parametermap = new HashMap<String, Double>();
		
		for (int index = 0; index < args.length; index ++) {
			if (index == 0)
				parametermap.put("learningRate", Double.valueOf(args[index]));
			else if (index == 1)
				parametermap.put("lamda", Double.valueOf(args[index]));
			else if (index == 2)
				parametermap.put("repetitions", Double.valueOf(args[index]));
		}
		
		return parametermap;
	}
	
	public static void main(String[] args) throws MLException {
		args = new String[5];
		args[0] = System.getProperty("user.dir") + "/src/main/resources/textClassification/data_set_1/";
		args[1] = "LR";
		args[2] = "0.01";
		args[3] = "0.01";
		args[4] = "100";
		
		if (args.length < 2) {
			System.out.println(" please provide valid inputs. kindly provide input in the following format :");
			System.out.println(" TextClassifierImpl <directory-location-containing-test-and-training-dirs> <classifier-name> <learning-rate> <lambda-value> <repetitions> ");
			System.out.println(" classifier-name 	: LR or NB");
			System.out.println(" learning-rate 		:  0.01");
			System.out.println(" lambda-value		:  0.8");
			System.out.println(" repetitions		:  500");
			System.out.println(" \t TextClassifierImpl ./data NB ");
			System.out.println(" \t TextClassifierImpl ./data LR ");
			System.out.println(" \t TextClassifierImpl ./data LR 0.01 0.8 200");
			System.out.println(" \t TextClassifierImpl ./data LR 0.01 0.9 500");
			System.out.println(" \t TextClassifierImpl ./data LR 0.5 0.8 500");
			System.exit(1);
		} 
		
		String dataDirectoryLocation = args[0];
		File baseDirectory = new File (dataDirectoryLocation);
		
		if (!(baseDirectory.exists() && baseDirectory.isDirectory())) {
			System.out.println(" please check the data directory location : " + dataDirectoryLocation);
			System.out.println(" either it does not exists, or if it exists, its not a directory (folder).");
			System.exit(1);
		}
		
		Classifier classifier = Classifier.valueOf(args[1]);
		if (classifier == null) {
			System.out.println(" invalid value for classifier specified. Please pass values only :");
			for (Classifier classifierItem : Classifier.values())
				System.out.println("  -  " + classifierItem.name());
			
			System.exit(1);
		}
		classifier.setBaseDirectory(baseDirectory);
		
		Map<String, Double> parameterMap = null;
		
		if (args.length > 2)
			parameterMap = parseArguments(Arrays.copyOfRange(args, 2, args.length));
		
		TextClassifier textClassifier;
		try {
			int loopIndex = 0;
			
			while (loopIndex ++ < 2) {
				textClassifier = new TextClassifier(baseDirectory, classifier);
				
				textClassifier.reset();
				
				if (parameterMap != null)
					classifier.setLearningParameters(parameterMap);
				
				boolean stopwordsUsed = (loopIndex % 2 == 0);
			
				textClassifier.setStopWordsUsed (stopwordsUsed);
				System.out.println(" -- running " + classifier.getClassifierName() + (stopwordsUsed ? " with " : " without ") + "stopwords ! ");

				textClassifier.train();
				double accuracy = textClassifier.test();
				
				System.out.println("\t-----------------------------");
				System.out.println("\ttotal accuracy : " + accuracy);
				
				System.out.println();
			}
		} catch (MLException e) {
			throw new MLException(" exception while performing text classification. ", e);
		}
	}
	
}
