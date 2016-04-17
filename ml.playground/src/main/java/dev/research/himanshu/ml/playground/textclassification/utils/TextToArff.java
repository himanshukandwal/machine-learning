package dev.research.himanshu.ml.playground.textclassification.utils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.StringTokenizer;

import dev.research.himanshu.ml.playground.decisiontree.model.MLException;
import dev.research.himanshu.ml.playground.textclassification.model.TextClass;
import dev.research.himanshu.ml.playground.textclassification.model.TextDocument;

/**
 * a utility class to convert test data (ham and spam) into weka arff files.
 * 
 * @author Himanshu Kandwal
 *
 */
public class TextToArff {

	public File baseDirectory;
	public Set<String> stopwords;
	private Map<TextClass, Map<String, Integer>> trainingClassifiedDictionary;
	private Map<TextClass, Map<String, Integer>> testClassifiedDictionary;
	private Map<TextClass, TextDocument[]> trainingClassifiedDocuments;
	private Map<TextClass, TextDocument[]> testClassifiedDocuments;
	private Set<String> trainingVocabulary;
	private Set<String> testVocabulary;
	private Set<String> globalVocabulary;

	public TextToArff(File baseDirectory) {
		this.baseDirectory = baseDirectory;
	}

	public Set<String> getStopwords() {
		return stopwords;
	}

	public void setStopwords(Set<String> stopwords) {
		this.stopwords = stopwords;
	}

	public Map<TextClass, Map<String, Integer>> getTrainingClassifiedDictionary() {
		if (trainingClassifiedDictionary == null)
			trainingClassifiedDictionary = new HashMap<TextClass, Map<String, Integer>>();

		return trainingClassifiedDictionary;
	}

	public void setTrainingClassifiedDictionary(Map<TextClass, Map<String, Integer>> trainingClassifiedDictionary) {
		this.trainingClassifiedDictionary = trainingClassifiedDictionary;
	}

	public Map<TextClass, Map<String, Integer>> getTestClassifiedDictionary() {
		if (testClassifiedDictionary == null)
			testClassifiedDictionary = new HashMap<TextClass, Map<String, Integer>>();

		return testClassifiedDictionary;
	}

	public void setTestClassifiedDictionary(Map<TextClass, Map<String, Integer>> testClassifiedDictionary) {
		this.testClassifiedDictionary = testClassifiedDictionary;
	}

	public Set<String> getTrainingVocabulary() {
		if (trainingVocabulary == null)
			trainingVocabulary = new LinkedHashSet<String>();

		return trainingVocabulary;
	}

	public Set<String> getTestVocabulary() {
		if (testVocabulary == null)
			testVocabulary = new LinkedHashSet<String>();

		return testVocabulary;
	}

	public Set<String> getGlobalVocabulary() {
		if (globalVocabulary == null)
			globalVocabulary = new LinkedHashSet<String>();

		return globalVocabulary;
	}
	
	public File getBaseDirectory() {
		return baseDirectory;
	}

	public void setBaseDirectory(File baseDirectory) {
		this.baseDirectory = baseDirectory;
	}
	
	public Map<TextClass, TextDocument[]> getTestClassifiedDocuments() {
		if (testClassifiedDocuments == null)
			testClassifiedDocuments = new HashMap<TextClass, TextDocument[]>();
			
		return testClassifiedDocuments;
	}
	
	public void setTestClassifiedDocuments(Map<TextClass, TextDocument[]> testClassifiedDocuments) {
		this.testClassifiedDocuments = testClassifiedDocuments;
	}

	public Map<TextClass, TextDocument[]> getTrainingClassifiedDocuments() {
		if (trainingClassifiedDocuments == null)
			trainingClassifiedDocuments = new HashMap<TextClass, TextDocument[]>();
			
		return trainingClassifiedDocuments;
	}
	
	public void setTrainingClassifiedDocuments(Map<TextClass, TextDocument[]> trainingClassifiedDocuments) {
		this.trainingClassifiedDocuments = trainingClassifiedDocuments;
	}

	public static void main(String[] args) throws IOException, MLException {
		if (args.length <= 0) {
			System.out.println("usage : please provide data directory path.");
			System.exit(1);
		}

		String dataDirectoryLocation = args[0];
		File baseDirectory = new File(dataDirectoryLocation);

		if (!(baseDirectory.exists() && baseDirectory.isDirectory())) {
			System.out.println(" please check the data directory location : " + dataDirectoryLocation);
			System.out.println(" either it does not exists, or if it exists, its not a directory (folder).");
			System.exit(1);
		}

		TextToArff converter = new TextToArff(baseDirectory);
		converter.loadContents();
		converter.generateArffs();
	}

	public void generateArffs() {
		generateFile(Constants.TRAIN);
		generateFile(Constants.TEST);
	}

	private void generateFile(String type) {
		BufferedWriter bufferedWriter = null;
		
		try {
			bufferedWriter = new BufferedWriter(new FileWriter(new File(getBaseDirectory(), type + Constants.ARFF_EXTENSION)));
			
			Map<TextClass, TextDocument[]> documentMap = (type == Constants.TRAIN ? getTrainingClassifiedDocuments() : getTestClassifiedDocuments());
			
			// write file header
			{
				bufferedWriter.append("@RELATION  'text-classification'");
				bufferedWriter.newLine();
	
				for (String classifiedVocabularyWord : getGlobalVocabulary()) {
					bufferedWriter.append("@ATTRIBUTE " + classifiedVocabularyWord + " NUMERIC");
					bufferedWriter.newLine();
				}
	
				bufferedWriter.append("@ATTRIBUTE @@class@@ {-1,1}");
				bufferedWriter.newLine();
				bufferedWriter.append("@DATA");
				bufferedWriter.newLine();
			}
	
			// write file data
			{
				for (Entry<TextClass, TextDocument[]> documentMapEntry : documentMap.entrySet()) {
					TextClass textClass = documentMapEntry.getKey();
					
					for (TextDocument textDocument : documentMapEntry.getValue()) {
						Map<String, Integer> textDocumentWordMap = populateDictionary(textDocument);
						StringBuffer stringBuffer = new StringBuffer();
						
						for (String classifiedVocabularyWord : getGlobalVocabulary()) 
							stringBuffer.append((textDocumentWordMap.containsKey(classifiedVocabularyWord) ? textDocumentWordMap.get(classifiedVocabularyWord) : 0) + ",");
						
						stringBuffer.append((textClass == TextClass.NEGATIVE_CLASS) ? 1 : -1);
						bufferedWriter.append(stringBuffer.toString());
						bufferedWriter.newLine();
					}
				}
			}
			
			bufferedWriter.close();
		} catch (Exception exception) {
			exception.printStackTrace();
		} finally {
			if (bufferedWriter != null)
				try {
					bufferedWriter.close();
				} catch (IOException e) {
					// consume exception.
				}
		}
	}

	/**
	 * load training and test data.
	 * @throws MLException 
	 */
	public void loadContents() throws MLException {
		// load stopwords.
		setStopwords(FileUtils.loadFile(baseDirectory, "stopwords.txt"));

		loadContents(Constants.TRAIN);
		loadContents(Constants.TEST);
	}

	public void loadContents(String type) throws MLException {
		/* load data */
		Map<TextClass, Map<String, Integer>> classifiedDictionary = (type == Constants.TRAIN ? getTrainingClassifiedDictionary() : getTestClassifiedDictionary());
		Set<String> classifiedVocabulary = (type == Constants.TRAIN ? getTrainingVocabulary() : getTestVocabulary());
		Map<TextClass, TextDocument[]> documentMap = (type == Constants.TRAIN ? getTrainingClassifiedDocuments() : getTestClassifiedDocuments());
		
		for (TextClass textClass : TextClass.values()) {
			TextDocument[] documents = FileUtils.prepareTextDocuments(FileUtils.locateClassData(baseDirectory, type, textClass), textClass);
			documentMap.put(textClass, documents);
			
			populateDictionary(classifiedDictionary, textClass, documents);
			classifiedVocabulary.addAll(classifiedDictionary.get(textClass).keySet());
		}
		
		getGlobalVocabulary().addAll(classifiedVocabulary);
	}

	protected Map<String, Integer> populateDictionary(Map<TextClass, Map<String, Integer>> dictionary,
			TextClass textClass, TextDocument[] instances) {
		return dictionary.put(textClass, populateDictionary(instances));
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

	public boolean isTokenValid(String token) {
		return ((!token.matches(Constants.NUMERICS)) && (!getStopwords().contains(token)) && (token.length() >= 2));
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
	
}