package dev.research.himanshu.ml.playground.imageclassification;

import java.io.File;

import org.junit.Test;

import junit.framework.TestCase;

public class KMeansTest extends TestCase {

	@Test
	public void testMainAll() throws Exception {
		String inputFolder = System.getProperty("user.dir") + "/src/main/resources/imageclassification/";
		String outputFolder = System.getProperty("user.dir") + "/src/main/resources/imageclassification/output/";
		
		for (File outputFolderFile :  new File(outputFolder).listFiles())
			outputFolderFile.delete();
		
		File [] inputfiles = new File(inputFolder).listFiles();
		for (File inputfile : inputfiles) {
			
			int[] kArr = { 2, 5, 10, 15, 20 };
			
			for (int k : kArr) {
				String [] arguments = new String [3];
				arguments [0] = inputfile.getAbsolutePath();
				arguments [1] = Integer.toString(k);
				arguments [2] = outputFolder + k + "-out-" + inputfile.getName();
				
				KMeans.main(arguments);
			}
		}
	}
	
	@Test
	public void testMainOne() throws Exception {
		String inputFolder = System.getProperty("user.dir") + "/src/main/resources/imageclassification/";
		String outputFolder = System.getProperty("user.dir") + "/src/main/resources/imageclassification/output/";
		
		int k = 2;
		
		for (File outputFolderFile :  new File(outputFolder).listFiles())
			if (outputFolderFile.getName().contains(Integer.toString(k) + "-"))
				outputFolderFile.delete();
		
		File [] inputfiles = new File(inputFolder).listFiles();

		for (File inputfile : inputfiles) {
			String [] arguments = new String [3];
			arguments [0] = inputfile.getAbsolutePath();
			arguments [1] = Integer.toString(k);
			arguments [2] = outputFolder + k + "-out-" + inputfile.getName();
			
			KMeans.main(arguments);
		}
	}
	
}
