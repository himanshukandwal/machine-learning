package dev.research.himanshu.ml.playground.textclassification.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.util.HashSet;
import java.util.Set;

import dev.research.himanshu.ml.playground.decisiontree.model.MLException;
import dev.research.himanshu.ml.playground.textclassification.model.TextClass;
import dev.research.himanshu.ml.playground.textclassification.model.TextDocument;

/**
 * class handling all the file system specific tasks.
 * 
 * @author Himanshu Kandwal
 *
 */
public class FileUtils {

	public static File[] locateClassData(File baseDirectory, String type, TextClass textClass) {
		File[] textClassFiles = null;
		File childClassDirectory = new File(baseDirectory, type + File.separator + textClass.getValue());
		
		if (childClassDirectory.exists() && childClassDirectory.isDirectory()) {
			textClassFiles = childClassDirectory.listFiles(new FilenameFilter() {

				public boolean accept(File dir, String name) {
					File file = new File(dir, name);
					if (file.isFile() && name.endsWith(Constants.EXTENSION))
						return true;
					return false;
				}

			});
		}

		return textClassFiles;
	}

	public static Set<String> loadFile(File baseDirectory, String filename) throws MLException {
		Set<String> lines = new HashSet<String>();
		try {
			BufferedReader bufferedReader = null;
			File textClassFile = new File(baseDirectory, filename);

			bufferedReader = new BufferedReader(new FileReader(textClassFile));
			String line = null;

			while ((line = bufferedReader.readLine()) != null)
				lines.add(line);

			bufferedReader.close();

		} catch (Exception exception) {
			throw new MLException(" caught in exception while loading text documents !");
		}

		return lines;
	}

	public static TextDocument[] prepareTextDocuments(File[] textClassFiles, TextClass textClass) throws MLException {
		TextDocument[] textDocuments = new TextDocument[textClassFiles.length];

		try {
			BufferedReader bufferedReader = null;
			for (int index = 0; index < textClassFiles.length; index ++) {
				File textClassFile = textClassFiles[index];

				bufferedReader = new BufferedReader(new FileReader(textClassFile));
				StringBuffer sb = new StringBuffer();
				String line = null;

				while ((line = bufferedReader.readLine()) != null) {
					sb.append(line).append(" ");
				}

				bufferedReader.close();

				textDocuments[index] = new TextDocument(textClass, sb.toString(), textClassFile.getAbsolutePath());
			}
		} catch (Exception exception) {
			throw new MLException(" caught in exception while loading text documents !");
		}

		return textDocuments;
	}

}
