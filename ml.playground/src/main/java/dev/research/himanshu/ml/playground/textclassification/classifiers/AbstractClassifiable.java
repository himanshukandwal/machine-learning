package dev.research.himanshu.ml.playground.textclassification.classifiers;

import java.io.File;
import java.util.HashMap;
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
	private Map<String, Integer> globalDictionary;
	private Map<TextClass, TextDocument[]> trainingInstances;
	private Map<TextClass, TextDocument[]> testInstances;
	private int totalTrainingInstances;
	private int totalTestInstances;
	private File baseDirectory;
	
	protected Set<String> getStopwords() {
		return stopwords;
	}
	
	protected void setStopwords(Set<String> stopwords) {
		this.stopwords = stopwords;
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
	
	protected Map<String, Integer> getGlobalDictionary() {
		if (null == globalDictionary)
			globalDictionary = new HashMap<String, Integer>();
		
		return globalDictionary;
	}

	protected Map<TextClass, TextDocument[]> getTrainingInstances() {
		if (null == trainingInstances) 
			trainingInstances = new HashMap<TextClass, TextDocument[]>();
			
		return trainingInstances;
	}
	
	public File getBaseDirectory() {
		return baseDirectory;
	}
	
	public void setBaseDirectory(File baseDirectory) {
		this.baseDirectory = baseDirectory;
	}
	
	public Map<TextClass, TextDocument[]> getTestInstances() {
		if (null == testInstances) 
			testInstances = new HashMap<TextClass, TextDocument[]>();
			
		return testInstances;
	}
	
	public void train() throws MLException {
		if (getBaseDirectory() == null)
			throw new MLException(" training not possible, as base directory has not been set yet. ");
		
		setStopwords(FileUtils.loadFile(getBaseDirectory(), "stopwords.txt"));
		
		/* load training data */
		for (TextClass textClass : TextClass.values()) {
			TextDocument[] documents = FileUtils.prepareTextDocuments(FileUtils.locateClassData (getBaseDirectory(), Constants.TRAIN, textClass), textClass);
			getTrainingInstances().put(textClass, documents);
			
			populateDictionary(documents);
			totalTrainingInstances += documents.length;
		}
		
		/* train it! */
		specificTraining();	
	}
	
	protected abstract void specificTraining();
	
	protected Map<String, Integer> populateDictionary(TextDocument[] instances) {
		return populateDictionary(instances, getGlobalDictionary());
	}

	protected Map<String, Integer> populateDictionary(TextDocument[] instances, Map<String, Integer> existingDictionary) {
		if (existingDictionary == null)
			existingDictionary = new HashMap<String, Integer>();
		
		StringTokenizer stringTokenizer;
		for (TextDocument textDocument : instances) {
			stringTokenizer = new StringTokenizer(textDocument.getContent(), Constants.DELIMITERS);
			
			while (stringTokenizer.hasMoreTokens()) {
				String token = stringTokenizer.nextToken();
				
				if (!getStopwords().contains(token)) {
					Integer oldVal = existingDictionary.put(token, 1);
					
					if (oldVal != null) 
						existingDictionary.put(token, ++ oldVal);
				}
			}
		}		
		
		return existingDictionary;
	}
	
	protected Map<String, Integer> populateDictionary(TextDocument instance) {
		Map<String, Integer> dictionary = new HashMap<String, Integer>();

		StringTokenizer stringTokenizer;
		stringTokenizer = new StringTokenizer(instance.getContent(), Constants.DELIMITERS);

		while (stringTokenizer.hasMoreTokens()) {
			String token = stringTokenizer.nextToken();

			if (!getStopwords().contains(token)) {
				Integer oldVal = dictionary.put(token, 1);

				if (oldVal != null)
					dictionary.put(token, ++oldVal);
			}
		}

		return dictionary;
	}
	
	public double test() throws MLException {
		/* load testing data */
		for (TextClass textClass : TextClass.values()) {
			TextDocument[] documents = FileUtils.prepareTextDocuments(FileUtils.locateClassData (getBaseDirectory(), Constants.TEST, textClass), textClass);
			getTestInstances().put(textClass, documents);
			totalTestInstances += documents.length;
		}
		
		/* train it! */
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
