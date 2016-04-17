package dev.research.himanshu.ml.playground.textclassification.classifiers;

import java.util.Arrays;
import java.util.Map;

import dev.research.himanshu.ml.playground.textclassification.model.TextClass;
import dev.research.himanshu.ml.playground.textclassification.model.TextDocument;

/**
 * class implementing Perceptron classifier.
 * 
 * @author Himanshu Kandwal
 * 
 */
public class PerceptronClassifier extends AbstractClassifiable {
	
	public Double learningRate = 0.01d;
	public int repetitions = 100;
	
	public double[][] classificationMatrix;
	public String[] classificationMetaHeader;
	public double[] weights;
	public double[] prior;
	
	public void setClassificationMatrix(double[][] classificationMatrix) {
		this.classificationMatrix = classificationMatrix;
	}
	
	public double[][] getClassificationMatrix() {
		return classificationMatrix;
	}
	
	public void setClassificationMetaHeader(String[] classificationMetaHeader) {
		this.classificationMetaHeader = classificationMetaHeader;
	}
	
	public String[] getClassificationMetaHeader() {
		return classificationMetaHeader;
	}
	
	public void setWeights(double[] weights) {
		this.weights = weights;
	}
	
	public double[] getWeights() {
		return weights;
	}
	
	public double[] getPrior() {
		return prior;
	}
	
	public void setPrior(double[] prior) {
		this.prior = prior;
	}
	
	@Override
	protected void specificTraining() {
		
		if (getLearningParameters() != null || getLearningParameters().size() > 0) {
			for (Map.Entry<String, Double> learningParameterMapEntry : getLearningParameters().entrySet()) {
				String parameterName = learningParameterMapEntry.getKey().toLowerCase();
			
				if (parameterName.equalsIgnoreCase("learningRate")) 
					learningRate = learningParameterMapEntry.getValue();
				else if (parameterName.equalsIgnoreCase("repetitions"))
					repetitions = learningParameterMapEntry.getValue().intValue();
			}
		}
		
		System.out.println("using learningRate : " + learningRate + ", repetitions : " + repetitions);
		
		// set the logistic regression specific variables
		setClassificationMetaHeader(getGlobalVocabulary().toArray(new String[0]));
		setClassificationMatrix(new double [getTotalTrainingInstances()][getClassificationMetaHeader().length + 2]);
		setWeights(new double[getClassificationMetaHeader().length + 1]);
		setPrior(new double[getTotalTrainingInstances()]);
		
		Arrays.fill (prior, (1d / getTotalTrainingInstances()));
		
		int classificationMatrixIndex = 0;
		for (TextClass textClass : TextClass.values()) {
			TextDocument[] textDocuments = getClassifiedTrainingInstances().get(textClass);
			
			for (TextDocument textDocument : textDocuments) {
				Map<String, Integer> localInstanceDictionary = populateDictionary (textDocument);
				
				// value for X0
				getClassificationMatrix() [classificationMatrixIndex][0] = 1;
				
				// values for Xi
				int columnIndex = 1;
				for (String classificationMetaHeaderValue : getClassificationMetaHeader())
					getClassificationMatrix() [classificationMatrixIndex][columnIndex ++] = 
						(localInstanceDictionary.get(classificationMetaHeaderValue) == null ? 0 : localInstanceDictionary.get(classificationMetaHeaderValue));
				
				// fill the classification result in the last column
				getClassificationMatrix() [classificationMatrixIndex][columnIndex] = ((textClass == TextClass.NEGATIVE_CLASS) ? 1 : -1);
				
				classificationMatrixIndex ++;
			}
		}
		
		// tune the weights.
		adjustWeights();
	}
	
	private void adjustWeights() {
		// initialize weights
		Arrays.fill(weights, 0.15);
		
		for (int iteration = 0; iteration < repetitions; iteration ++) {
			for (int row = 0; row < getTotalTrainingInstances(); row ++) {
				
				double stepVal = weights[0];
				double deltaWeights[] = new double[getClassificationMetaHeader().length + 1];
				
				// calculating WX = [sum (Wi * Xi)]
				for (int column = 1; column < getClassificationMetaHeader().length + 1; column ++)
					stepVal += weights[column] * getClassificationMatrix()[row][column];
				
				prior[row] = stepVal > 0 ? 1 : -1;

				for (int k = 0; k < getClassificationMetaHeader().length + 1; k++) {
					deltaWeights[k] += getClassificationMatrix()[row][k]  
							* (getClassificationMatrix()[row][getClassificationMetaHeader().length + 1] - prior[row]);
				
					weights[k] += learningRate * (deltaWeights[k]);
				}
			}
		}
	}

	@Override
	protected double specificTesting() {
		int totalCorrectClassifications = 0;
		for (TextClass textClass : TextClass.values()) {
			
			int correctClassifications = 0;
			TextDocument[] classTextDocuments = getClassifiedTestInstances().get(textClass);
			
			for (TextDocument classTextDocument : classTextDocuments) {
				Map<String, Integer> localInstanceDictionary = populateDictionary(classTextDocument);
				
				double[] predictions = new double[getClassificationMetaHeader().length + 2];
				
				// value for X0
				predictions[0] = 1;
				
				// values for Xi
				int columnIndex = 1;
				for (String classificationMetaHeaderValue : getClassificationMetaHeader())
					predictions [columnIndex ++] = (localInstanceDictionary.get(classificationMetaHeaderValue) == null ? 
							0 : localInstanceDictionary.get(classificationMetaHeaderValue));
				
				// compute overall output prediction.
				double output = weights[0];
				
				for (int index = 1; index < getClassificationMetaHeader().length + 1; index ++)
					output += weights [index] * predictions [index];
					
				if (output > 0)
					correctClassifications ++;				
			}
			
			if (textClass == TextClass.POSITIVE_CLASS) 
				correctClassifications = (classTextDocuments.length - correctClassifications);
			
			totalCorrectClassifications += correctClassifications;
			System.out.println("\t" + textClass.getValue() + " accuracy : " + (double) (correctClassifications * 100d / classTextDocuments.length));
		}
		
		return (double) (totalCorrectClassifications * 100d / getTotalTestInstances());
	}
		
}
