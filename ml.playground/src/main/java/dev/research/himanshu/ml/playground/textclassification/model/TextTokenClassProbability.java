package dev.research.himanshu.ml.playground.textclassification.model;

/**
 * model class presenting the data structure, in which the probabilities of the token of the instances will be stored. 
 * 
 * @author Himanshu Kandwal
 *
 */
public class TextTokenClassProbability {
	
	private String token;
	private TextClass textClass;
	private Double conditionalProbability;
	
	public TextTokenClassProbability(String token, TextClass textClass) {
		super();
		this.token = token;
		this.textClass = textClass;
	}
	
	public TextTokenClassProbability(String token, TextClass textClass, Double conditionalProbability) {
		this(token, textClass);
		this.conditionalProbability = conditionalProbability;
	}
	
	public String getToken() {
		return token;
	}
	
	public void setToken(String token) {
		this.token = token;
	}
	
	public TextClass getTextClass() {
		return textClass;
	}
	
	public void setTextClass(TextClass textClass) {
		this.textClass = textClass;
	}
	
	public Double getConditionalProbability() {
		return conditionalProbability;
	}
	
	public void setConditionalProbability(Double conditionalProbability) {
		this.conditionalProbability = conditionalProbability;
	}
	
}
