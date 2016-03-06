package dev.research.himanshu.ml.playground.textclassification.model;

/**
 * enumeration class having the valid text class names. 
 * 
 * @author Himanshu Kandwal
 *
 */
public enum TextClass {
	
	POSITIVE_CLASS ("ham"),
	NEGATIVE_CLASS ("spam");
	
	private String value;
	
	private TextClass (String value) {
		this.value = value;
	}
	
	public String getValue() {
		return value;
	}

}

