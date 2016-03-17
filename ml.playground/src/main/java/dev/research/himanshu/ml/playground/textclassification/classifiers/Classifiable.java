package dev.research.himanshu.ml.playground.textclassification.classifiers;

import java.io.File;
import java.util.Map;

import dev.research.himanshu.ml.playground.decisiontree.model.MLException;

/**
 * generic interface presenting a simple classifiable interface. 
 * 
 * @author Himanshu Kandwal
 *
 */
public interface Classifiable {
	
	public void reset();
	
	public void setLearningParameters(Map<String, Double> learningParameters);
	
	public void setBaseDirectory(File baseDirectory);
	
	public void train(boolean stopWordsUsed) throws MLException;
	
	public double test() throws MLException;

}
