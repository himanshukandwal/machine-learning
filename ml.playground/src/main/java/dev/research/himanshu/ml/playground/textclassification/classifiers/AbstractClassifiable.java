package dev.research.himanshu.ml.playground.textclassification.classifiers;

import java.io.File;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;

import dev.research.himanshu.ml.playground.decisiontree.model.MLException;
import dev.research.himanshu.ml.playground.textclassification.model.TextClass;
import dev.research.himanshu.ml.playground.textclassification.model.TextDocument;
import dev.research.himanshu.ml.playground.textclassification.utils.Constants;
import dev.research.himanshu.ml.playground.textclassification.utils.FileUtils;

/**
 * abstract class for handling the background shared code
 * 
 * @author Himanshu Kandwal
 *
 */
public abstract class AbstractClassifiable implements Classifiable {

	private Set<String> stopwords;
	private Map<TextClass, Map<String, Integer>> globalClassifiedDictionary;
	private Map<TextClass, TextDocument[]> classifiedTrainingInstances;
	private Map<TextClass, TextDocument[]> classifiedTestInstances;
	private Set<String> globalVocabulary;
	private int totalTrainingInstances;
	private int totalTestInstances;
	private File baseDirectory;
	private Map<String, Double> learningParameters;
	protected MathContext divisionMathContext = new MathContext(5, RoundingMode.HALF_EVEN);
	
	protected Set<String> getStopwords() {
		if (this.stopwords == null)
			this.stopwords = new LinkedHashSet<String>();
		return stopwords;
	}
	
	public void setLearningParameters(Map<String, Double> learningParameters) {
		this.learningParameters = learningParameters;
	}
	
	public Map<String, Double> getLearningParameters() {
		if (null == learningParameters)
			learningParameters = new LinkedHashMap<String, Double>();

		return learningParameters;
	}
	
	protected void addStopwords(Set<String> stopwords) {
		this.stopwords.addAll(stopwords);
	}
	
	protected int getTotalTrainingInstances() {
		return totalTrainingInstances;
	}
	
	protected void setTotalTrainingInstances(int totalInstances) {
		this.totalTrainingInstances = totalInstances;
	}
	
	public int getTotalTestInstances() {
		return totalTestInstances;
	}
	
	public void setTotalTestInstances(int totalTestInstances) {
		this.totalTestInstances = totalTestInstances;
	}
	
	protected Map<TextClass, Map<String, Integer>> getGlobalClassifiedDictionary() {
		if (null == globalClassifiedDictionary)
			globalClassifiedDictionary = new LinkedHashMap<TextClass, Map<String, Integer>>();
		
		return globalClassifiedDictionary;
	}

	protected Map<TextClass, TextDocument[]> getClassifiedTrainingInstances() {
		if (null == classifiedTrainingInstances) 
			classifiedTrainingInstances = new LinkedHashMap<TextClass, TextDocument[]>();
			
		return classifiedTrainingInstances;
	}
	
	public File getBaseDirectory() {
		return baseDirectory;
	}
	
	public void setBaseDirectory(File baseDirectory) {
		this.baseDirectory = baseDirectory;
	}
	
	public Map<TextClass, TextDocument[]> getClassifiedTestInstances() {
		if (null == classifiedTestInstances) 
			classifiedTestInstances = new LinkedHashMap<TextClass, TextDocument[]>();
			
		return classifiedTestInstances;
	}
	
	public Set<String> getGlobalVocabulary() {
		if (globalVocabulary == null)
			globalVocabulary = new LinkedHashSet<String>();
		
		return globalVocabulary;
	}
	
	public void reset() {
		getClassifiedTestInstances().clear();
		getClassifiedTrainingInstances().clear();
		getGlobalClassifiedDictionary().clear();
		getGlobalVocabulary().clear();
		getLearningParameters().clear();
		getStopwords().clear();
		setTotalTestInstances(0);
		setTotalTrainingInstances(0);
	}
	
	public void train(boolean stopWordsUsed) throws MLException {
		if (getBaseDirectory() == null)
			throw new MLException(" training not possible, as base directory has not been set yet. ");
		
		if (stopWordsUsed) {
			addStopwords(FileUtils.loadFile(getBaseDirectory(), "stopwords.txt"));
		}
		
		/* load training data */
		for (TextClass textClass : TextClass.values()) {
			TextDocument[] documents = FileUtils.prepareTextDocuments(FileUtils.locateClassData (getBaseDirectory(), Constants.TRAIN, textClass), textClass);
			getClassifiedTrainingInstances().put(textClass, documents);
			
			populateDictionary(textClass, documents);
			getGlobalVocabulary().addAll(getGlobalClassifiedDictionary().get(textClass).keySet());
			
			totalTrainingInstances += documents.length;
		}
		
		manageClassDictionaries();
		
		/* train it! */
		specificTraining();	
	}

	protected abstract void specificTraining();
	
	protected Map<String, Integer> populateDictionary(TextClass textClass, TextDocument[] instances) {
		return getGlobalClassifiedDictionary().put(textClass, populateDictionary(instances));
	}
	
	private void manageClassDictionaries() {
		for (TextClass textClass : TextClass.values()) {
			Map<String, Integer> sourceClassDictionaryMap = getGlobalClassifiedDictionary().get(textClass);
			
			for (TextClass innerTextClass : TextClass.values()) {
				if (innerTextClass == textClass)
					continue;
				else {
					Map<String, Integer> targetClassDictionaryMap = getGlobalClassifiedDictionary().get(innerTextClass);
		
					for (Map.Entry<String, Integer> sourceClassDictionaryMapEntry : sourceClassDictionaryMap.entrySet()) {
						if (!targetClassDictionaryMap.containsKey(sourceClassDictionaryMapEntry.getKey()))
							targetClassDictionaryMap.put(sourceClassDictionaryMapEntry.getKey(), 0);
					}
				}
			}
		}
	}

	protected Map<String, Integer> populateDictionary(TextDocument[] instances) {
		return populateDictionary(instances, null);
	}

	protected Map<String, Integer> populateDictionary(TextDocument[] instances, Map<String, Integer> existingDictionary) {
		if (existingDictionary == null)
			existingDictionary = new LinkedHashMap<String, Integer>();
		
		StringTokenizer stringTokenizer;
		for (TextDocument textDocument : instances) {
			stringTokenizer = new StringTokenizer(textDocument.getContent(), Constants.DELIMITERS);
			
			while (stringTokenizer.hasMoreTokens()) {
				String token = stringTokenizer.nextToken().trim();
				
				if (isTokenValid(token)) {
					if (existingDictionary.containsKey(token))
						existingDictionary.put(token, existingDictionary.get(token) + 1);
					else
						existingDictionary.put(token, 1);
				}
			}
		}		
		
		return existingDictionary;
	}
	
	protected Map<String, Integer> populateDictionary(TextDocument instance) {
		Map<String, Integer> dictionary = new LinkedHashMap<String, Integer>();

		StringTokenizer stringTokenizer;
		stringTokenizer = new StringTokenizer(instance.getContent(), Constants.DELIMITERS);

		while (stringTokenizer.hasMoreTokens()) {
			String token = stringTokenizer.nextToken().trim();
			
			if (isTokenValid(token)) {
				if (dictionary.containsKey(token))
					dictionary.put(token, dictionary.get(token) + 1);
				else
					dictionary.put(token, 1);
			}
		}
		
		return dictionary;
	}
	
	public boolean isTokenValid(String token) {
		return ((!token.matches(Constants.NUMERICS)) && (!getStopwords().contains(token)) && (token.length() >= 2));
	}
	
	public double test() throws MLException {
		/* load testing data */
		for (TextClass textClass : TextClass.values()) {
			TextDocument[] documents = FileUtils.prepareTextDocuments(FileUtils.locateClassData (getBaseDirectory(), Constants.TEST, textClass), textClass);
			getClassifiedTestInstances().put(textClass, documents);
			totalTestInstances += documents.length;
		}
		
		/* test it! */
		return specificTesting();	
	}
	
	protected abstract double specificTesting();
	
	protected int aggregateTokenCount(Map<String, Integer> dictionary) {
		int totalcount = 0;
		for (Map.Entry<String, Integer> dictionaryEntry : dictionary.entrySet())
			totalcount += dictionaryEntry.getValue();
		
		return totalcount;
	}
		
}
