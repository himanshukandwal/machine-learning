package dev.research.himanshu.ml.playground.textclassification;

import java.io.File;

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
	
	public void train() throws MLException {
		try {
			/** perform training **/
			getClassifier().getClassifiable().train();
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
	
	public static void main(String[] args) throws MLException {
		args = new String[2];
		args[0] = System.getProperty("user.dir") + "/src/main/resources/textClassification/";
		args[1] = "NB";
		
		if (args.length < 2) {
			System.out.println(" please provide valid inputs. kindly provide input in the following format :");
			System.out.println(" TextClassifierImpl <directory-location-containing-test-and-training-dirs> <classifier-name>");
			System.out.println(" classifier-name : LR or NB");
			System.out.println(" example : ");
			System.out.println(" \t TextClassifierImpl ./data LR ");
			System.out.println(" \t TextClassifierImpl ./data NB ");
			
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
		
		
		TextClassifier textClassifier = new TextClassifier(baseDirectory, classifier);
		try {
			textClassifier.train();
			
			double accuracy = textClassifier.test();
			
			System.out.println(" finsihed evaluation of test data using " + classifier.getClassifierName() + " with accuracy : " + accuracy);
			
		} catch (MLException e) {
			throw new MLException(" exception while performing text classification. ", e);
		}
	}
	
}
