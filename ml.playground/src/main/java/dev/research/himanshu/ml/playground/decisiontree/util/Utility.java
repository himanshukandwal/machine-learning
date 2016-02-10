package dev.research.himanshu.ml.playground.decisiontree.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import dev.research.himanshu.ml.playground.decisiontree.model.AttributeDO;
import dev.research.himanshu.ml.playground.decisiontree.model.Instance;
import dev.research.himanshu.ml.playground.decisiontree.model.Instances;
import dev.research.himanshu.ml.playground.decisiontree.model.MLException;

/**
 * A utility class for handling all the extra activities.
 * 
 * @author Himanshu Kandwal
 */
public class Utility {
	
	public static List<String> loadFile (String filelocation) throws MLException {
		List<String> lines = null;
		File file = new File(filelocation);
		
		if (file.exists()) {
			BufferedReader bufferedReader = null;
			try {
				bufferedReader = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
				String line = null;
				lines = new ArrayList<String>();
				
				while ((line = bufferedReader.readLine()) != null) {
					if (line.isEmpty())
						continue;
					
					lines.add(line);
				}
			} catch (FileNotFoundException exception) {
				throw new MLException("FileNotFoundException !", exception);
			} catch (IOException exception) {
				throw new MLException("IOException !", exception);
			} finally {
				if (bufferedReader != null) {
					try { bufferedReader.close(); } catch (IOException e) { /* consume exception */ }
				}
			}
		}
		return lines;
	}
	
	public static Instances loadInstancesFromData (List<String> sources) throws MLException {
		Instance[] outcomes = new Instance [sources.size() - 1];
		Instances instances = new Instances(outcomes);
		try {
			Map<String, Integer> metadata = new HashMap<String, Integer>();
			Map<Integer, String> positionValueMap = new HashMap<Integer, String>();
			
			int index = 0;
			for (String header : sources.get(0).split(",")) {
				metadata.put(header, index);
				positionValueMap.put(index ++, header);
			}
			
			instances.setHeader(metadata);
			
			for (index = 1; index < sources.size(); index ++) {
				AttributeDO[] attributes = new AttributeDO[metadata.size()];
				String [] source = sources.get(index).split(",");
				
				for (int loop = 0; loop < source.length; loop ++)
					attributes[loop] = new AttributeDO(positionValueMap.get(loop), Integer.valueOf(source [loop]));
				
				outcomes[index - 1] = new Instance(attributes);
			}
			
			positionValueMap.clear();
		} catch (NumberFormatException exception) {
			throw new MLException(" NumberFormatException !", exception);
		}
		return instances;		
	}
	
}
