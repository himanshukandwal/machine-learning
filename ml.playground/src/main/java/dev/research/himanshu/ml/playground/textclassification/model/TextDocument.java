package dev.research.himanshu.ml.playground.textclassification.model;

/**
 * model class presenting the data structure, in which the content of the instances will be stored. 
 * 
 * @author Himanshu Kandwal
 *
 */
public class TextDocument {
	
	private TextClass textClass;
	private String content;
	private String documentPath;
	
	public TextDocument(TextClass textClass) {
		this.textClass = textClass;
	}

	public TextDocument(TextClass textClass, String content) {
		this(textClass);
		this.content = content;
	}

	public TextDocument(TextClass textClass, String content, String documentPath) {
		this(textClass, content);
		this.documentPath = documentPath;
	}
	
	public String getContent() {
		return content;
	}

	public void setContent(String content) {
		this.content = content;
	}
	
	public TextClass getTextClass() {
		return textClass;
	}
	
	public void setTextClass(TextClass textClass) {
		this.textClass = textClass;
	}
	
	public void setDocumentPath(String documentPath) {
		this.documentPath = documentPath;
	}
	
	public String getDocumentPath() {
		return documentPath;
	}
	
	@Override
	public String toString() {
		return textClass.getValue().toString();
	}
	
}
