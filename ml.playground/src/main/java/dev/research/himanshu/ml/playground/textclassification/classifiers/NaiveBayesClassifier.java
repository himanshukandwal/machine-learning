package dev.research.himanshu.ml.playground.textclassification.classifiers;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import dev.research.himanshu.ml.playground.textclassification.model.TextClass;
import dev.research.himanshu.ml.playground.textclassification.model.TextDocument;

/**
 * class implementing Naive Bayes classifier.
 * 
 * @author Himanshu Kandwal
 * 
 */
public class NaiveBayesClassifier extends AbstractClassifiable {

	private Map<TextClass, Double> classPriorMap;
	private Map<String, Map<TextClass, Double>> classifiedProbabilityMap;

	@Override
	public void reset() {
		super.reset();
		getClassPriorMap().clear();
		getClassifiedProbabilityMap().clear();
	}
	
	public Map<TextClass, Double> getClassPriorMap() {
		if (null == classPriorMap)
			classPriorMap = new LinkedHashMap<TextClass, Double>();

		return classPriorMap;
	}

	public Map<String, Map<TextClass, Double>> getClassifiedProbabilityMap() {
		if (classifiedProbabilityMap == null)
			classifiedProbabilityMap = new LinkedHashMap<String, Map<TextClass, Double>>();

		return classifiedProbabilityMap;
	}

	@Override
	public void specificTraining() {

		for (TextClass textClass : TextClass.values()) {
			
			// calculate class prior
			getClassPriorMap().put(textClass, (double) getClassifiedTrainingInstances().get(textClass).length / getTotalTrainingInstances());

			// creating denominator for conditional probability.
			long commonDenominator = aggregateTokenCount(getGlobalClassifiedDictionary().get(textClass))
					+ getGlobalVocabulary().size();
			
			for (Map.Entry<String, Integer> localClassifiedDictionaryEntry : getGlobalClassifiedDictionary().get(textClass).entrySet()) {
				String token = localClassifiedDictionaryEntry.getKey();
				
				// creating numerator for conditional probability.
				long numerator = localClassifiedDictionaryEntry.getValue() + 1;
				
				double fraction = (double) numerator / commonDenominator;

				if (!getClassifiedProbabilityMap().containsKey(token)) {
					Map<TextClass, Double> classifiedProbability = new HashMap<TextClass, Double>();
					classifiedProbability.put(textClass, fraction);

					getClassifiedProbabilityMap().put(token, classifiedProbability);
				} else {
					Map<TextClass, Double> classifiedProbability = getClassifiedProbabilityMap().get(token);
					classifiedProbability.put(textClass, fraction);
				}
			}
		}
	}

	@Override
	protected double specificTesting() {
		int totalCorrectClassifications = 0;
		
		for (Map.Entry<TextClass, TextDocument[]> getClassifiedTestInstancesEntry : getClassifiedTestInstances().entrySet()) {
			
			TextClass originalTextClass = getClassifiedTestInstancesEntry.getKey();
			TextDocument[] classTextDocuments = getClassifiedTestInstancesEntry.getValue();
			int correctClassifications = 0;
			
			for (TextDocument textDocument : classTextDocuments) {
				Map<TextClass, Double> classificationDataMap = new HashMap<TextClass, Double>();

				// instance specific dictionary
				Map<String, Integer> localInstanceDictionary = populateDictionary(textDocument);
				
				for (TextClass textClass : TextClass.values()) {
					Double overallInstanceClassficationValue = (Math.log (getClassPriorMap().get(textClass)) / Math.log(2));

					for (Map.Entry<String, Integer> localInstanceDictionaryEntry : localInstanceDictionary.entrySet()) {
						String token = localInstanceDictionaryEntry.getKey();
						Double classifiedConditionalProbablity;

						if (getClassifiedProbabilityMap().containsKey(token)) {
							classifiedConditionalProbablity = getClassifiedProbabilityMap().get(token).get(textClass);
							overallInstanceClassficationValue += (double) localInstanceDictionaryEntry.getValue() * ((Math.log (classifiedConditionalProbablity) / Math.log(2)));
						}
					}

					classificationDataMap.put(textClass, overallInstanceClassficationValue);
				}
				
				// best class for the test instance
				Double maxCount = null;
				TextClass bestClass = null;

				for (Map.Entry<TextClass, Double> classificationDataEntry : classificationDataMap.entrySet()) {
					if (maxCount == null || (classificationDataEntry.getValue().compareTo(maxCount) > 0)) {
						maxCount = classificationDataEntry.getValue();
						bestClass = classificationDataEntry.getKey();
					}
				}

				if (originalTextClass == bestClass)
					correctClassifications ++;
			}
			totalCorrectClassifications += correctClassifications;
			System.out.println("\t" + originalTextClass.getValue() + " accuracy : " + (double) (correctClassifications * 100d / classTextDocuments.length));
		}
		
		return (double) (totalCorrectClassifications * 100d / getTotalTestInstances());
	}

}