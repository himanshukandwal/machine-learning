package dev.research.himanshu.ml.playground.textclassification.classifiers;

import java.io.File;

import dev.research.himanshu.ml.playground.decisiontree.model.MLException;

/**
 * generic interface presenting a simple classifiable interface. 
 * 
 * @author Himanshu Kandwal
 *
 */
public interface Classifiable {
	
	public void setBaseDirectory(File baseDirectory);
	
	public void train() throws MLException;
	
	public double test() throws MLException;

}
