package dev.research.himanshu.ml.playground.textclassification.classifiers;

import java.util.Arrays;
import java.util.Map;

import dev.research.himanshu.ml.playground.textclassification.model.TextClass;
import dev.research.himanshu.ml.playground.textclassification.model.TextDocument;

/**
 * Logic implementing logistic regression for text classification
 * 
 * @author Himanshu Kandwal
 *
 */
public class LogisticRegression extends AbstractClassifiable {
	
	public Double learningRate = 0.01;
	public Double lamda = 0.8;
	public int repetitions = 50;
	
	public double[][] classificationMatrix;
	public String[] classificationMetaHeader;
	public double[] weights;
	
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
	
	@Override
	protected void specificTraining() {
		
		if (getLearningParameters() != null || getLearningParameters().size() > 0) {
			for (Map.Entry<String, Double> learningParameterMapEntry : getLearningParameters().entrySet()) {
				String parameterName = learningParameterMapEntry.getKey().toLowerCase();
				
				if (parameterName.equalsIgnoreCase("learningRate")) 
					learningRate = learningParameterMapEntry.getValue();
				else if (parameterName.equalsIgnoreCase("lamda"))
					lamda = learningParameterMapEntry.getValue();
				else if (parameterName.equalsIgnoreCase("repetitions"))
					repetitions = learningParameterMapEntry.getValue().intValue();
			}
		}
		
		System.out.println("using learningRate : " + learningRate + ", lamda : " + lamda + ", repetitions : " + repetitions);
		
		// set the logistic regression specific variables
		setClassificationMetaHeader(getGlobalVocabulary().toArray(new String[0]));
		setClassificationMatrix(new double [getTotalTrainingInstances()][getClassificationMetaHeader().length + 2]);
		setWeights(new double[getGlobalVocabulary().size() + 2]);
		
		int classificationMatrixIndex = 0;
		for (TextClass textClass : TextClass.values()) {
			TextDocument[] textDocuments = getClassifiedTrainingInstances().get(textClass);
			
			for (TextDocument textDocument : textDocuments) {
				Map<String, Integer> localInstanceDictionary = populateDictionary (textDocument);
				
				// value for x0
				getClassificationMatrix() [classificationMatrixIndex][0] = 1;
				
				// values for Xi
				int columnIndex = 1;
				for (String classificationMetaHeaderValue : getClassificationMetaHeader())
					getClassificationMatrix() [classificationMatrixIndex][columnIndex ++] = 
						(localInstanceDictionary.get(classificationMetaHeaderValue) == null ? 0 : localInstanceDictionary.get(classificationMetaHeaderValue));
				
				// fill the classification result in the last column
				getClassificationMatrix() [classificationMatrixIndex][columnIndex] = ((textClass == TextClass.NEGATIVE_CLASS) ? 1 : 0);
				
				classificationMatrixIndex ++;
			}
		}

		adjustWeights();
	}
	
	private void adjustWeights() {
		// initialize weights
		Arrays.fill(getWeights(), 0.001);
		Double[] activationUnits = null;
		
		for (int iteration = 0; iteration < repetitions; iteration ++) {
			activationUnits = new Double [getClassificationMatrix().length];
			
			for (int row = 0; row < getClassificationMatrix().length; row ++) {
				Double activationValue = 0d;
				
				for (int column = 0; column < getGlobalVocabulary().size() + 1; column ++) {
					activationValue += getWeights()[column] * getClassificationMatrix()[row][column];
				}
				
				// estimating values between [0 - 1]
				activationUnits[row] = sigmoidFunction (activationValue, TextClass.NEGATIVE_CLASS);
			}

			for (int column = 0; column < getGlobalVocabulary().size() + 1; column ++) {
				Double originalWeight = getWeights()[column];

				Double increment = 0.0;
				for (int row = 0; row < getTotalTrainingInstances(); row ++) {
					double inputCount = getClassificationMatrix()[row][column];

					double outputClass = getClassificationMatrix()[row][getGlobalVocabulary().size() + 1];

					Double activationUnit = activationUnits[row];

					increment = increment + inputCount * (outputClass - activationUnit);
				}
				
				increment = increment - lamda * originalWeight;
				Double newWeight = originalWeight + learningRate * increment;
				getWeights()[column] = newWeight;
			}
		}
	}
	
	public Double sigmoidFunction(Double activationValue, TextClass textClass) {
		Double denom = 1 + Math.exp(activationValue);

		Double output = 1 / denom;
		if (textClass == TextClass.NEGATIVE_CLASS)
			output = 1 - output;
		
		return output;
	}

	@Override
	protected double specificTesting() {
		int totalCorrectClassifications = 0;
		for (TextClass textClass : TextClass.values()) {
			
			int correctClassifications = 0;
			TextDocument[] classTextDocuments = getClassifiedTestInstances().get(textClass);
			
			for (TextDocument classTextDocument : classTextDocuments) {
				Map<String, Integer> localInstanceDictionary = populateDictionary(classTextDocument);
				
				double prediction = getWeights()[0];
				for (int columnIndex = 1; columnIndex < getGlobalVocabulary().size() + 1; columnIndex ++) {
					prediction += getWeights()[columnIndex] * (localInstanceDictionary.get(getClassificationMetaHeader()[columnIndex -1]) == null ? 
							0 : localInstanceDictionary.get(getClassificationMetaHeader()[columnIndex -1]));
				}	
			
				Double output = sigmoidFunction(prediction, textClass);
				
				if (output > 0.5)
					correctClassifications ++;				
			}
			
			totalCorrectClassifications += correctClassifications;
			System.out.println("\t" + textClass.getValue() + " accuracy : " + (double) (correctClassifications * 100d / classTextDocuments.length));
		}
		
		return (double) (totalCorrectClassifications * 100d / getTotalTestInstances());
	}
		
}
