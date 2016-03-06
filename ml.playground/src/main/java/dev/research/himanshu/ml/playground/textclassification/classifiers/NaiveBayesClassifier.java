package dev.research.himanshu.ml.playground.textclassification.classifiers;

import java.util.HashMap;
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
	
	public Map<TextClass, Double> getClassPriorMap() {
		if (null == classPriorMap)
			classPriorMap = new HashMap<TextClass, Double>();
		
		return classPriorMap;
	}
	
	public Map<String, Map<TextClass, Double>> getClassifiedProbabilityMap() {
		if (classifiedProbabilityMap == null)
			classifiedProbabilityMap = new HashMap<String, Map<TextClass, Double>>();
		
		return classifiedProbabilityMap;
	}

	@Override
	public void specificTraining() {
		
		for (TextClass textClass : TextClass.values()) {
			TextDocument[] classTextDocuments = getTrainingInstances().get(textClass);
			int numberOfDocumentsInClass = classTextDocuments.length;
			
			double prior = (double) numberOfDocumentsInClass / getTotalTrainingInstances();
			getClassPriorMap().put(textClass, prior);
			
			// class specific dictionary
			Map<String, Integer> classSpecificDictionary = populateDictionary (classTextDocuments);
			
			// creating denominator for conditional probability.
		    long denominator = aggregateTokenCount (classSpecificDictionary) + getGlobalDictionary().size();
		    
		    for (Map.Entry<String, Integer> globalDictionaryEntry : getGlobalDictionary().entrySet()) {
				String token = globalDictionaryEntry.getKey();
				
				int localDictionaryCount = (classSpecificDictionary.get(token) == null ? 0 : classSpecificDictionary.get(token));
				
				// creating numerator for conditional probability.
			    float numerator = localDictionaryCount + 1.0f;
			    
			    double conditionalProbability = numerator / denominator;
			    
			    if (!getClassifiedProbabilityMap().containsKey(token)) {
			    	Map<TextClass, Double> classifiedProbability = new HashMap<TextClass, Double>();
			    	classifiedProbability.put(textClass, conditionalProbability);
			    	
			    	getClassifiedProbabilityMap().put(token, classifiedProbability);
				} else {
					Map<TextClass, Double> classifiedProbability = getClassifiedProbabilityMap().get(token);
					classifiedProbability.put(textClass, conditionalProbability);
				}
		    }
		}
	}

	@Override
	protected double specificTesting() {
		
		int correctClassifications = 0;
		
		for (TextClass textClass : TextClass.values()) {
			TextDocument[] classTextDocuments = getTrainingInstances().get(textClass);
			
			for (TextDocument testInstanceClassTextDocuments : classTextDocuments) {
				Map<TextClass, Double> classificationCountMap = new HashMap<TextClass, Double>();
				
				// class specific dictionary
				Map<String, Integer> classSpecificDictionary = populateDictionary (testInstanceClassTextDocuments);
				for (TextClass localTextClass : TextClass.values()) {
					Double textClassCount = Math.log (getClassPriorMap().get(localTextClass));
					
					for (Map.Entry<String, Integer> classSpecificDictionaryEntry : classSpecificDictionary.entrySet()) {
						String classToken = classSpecificDictionaryEntry.getKey();
						
						Map<TextClass, Double> classifiedProbability = getClassifiedProbabilityMap().get(classToken);
						textClassCount += ((classifiedProbability == null) ? 0 : Math.log (classifiedProbability.get(textClass)));
					}
					
					classificationCountMap.put(localTextClass, textClassCount);
				}
				
				// best class for the test instance
				Double maxCount = Double.MIN_VALUE;
				TextClass bestClass = null;
				
				for (Map.Entry<TextClass, Double> classificationCountMapEntry : classificationCountMap.entrySet()) {
					if (classificationCountMapEntry.getValue() > maxCount) {
						maxCount = classificationCountMapEntry.getValue();
						bestClass = classificationCountMapEntry.getKey();
					}
				}
				
				if (textClass == bestClass) {
					correctClassifications ++;
				}
			}
		}
		return correctClassifications / getTotalTestInstances();
	}
	
}
