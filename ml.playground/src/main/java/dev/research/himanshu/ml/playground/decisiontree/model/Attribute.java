package dev.research.himanshu.ml.playground.decisiontree.model;

import java.math.BigDecimal;
import java.util.Map;

public class Attribute {

	private String attributeName;
	private Map<Integer, AttributeValue> attributeValues;
	private BigDecimal entropy;
	private BigDecimal informationGain;
	
	public Attribute() {}

	public Attribute(String attributeName, Map<Integer, AttributeValue> attributeValue) {
		this();
		this.attributeName = attributeName;
		this.attributeValues = attributeValue;
	}
	
	public Attribute(String attributeName, Map<Integer, AttributeValue> attributeValue, BigDecimal entropy) {
		this(attributeName, attributeValue);
		this.entropy = entropy;
	}

	public String getAttributeName() {
		return attributeName;
	}

	public void setAttributeName(String attributeName) {
		this.attributeName = attributeName;
	}

	public Map<Integer, AttributeValue> getAttributeValues() {
		return attributeValues;
	}

	public void setAttributeValues(Map<Integer, AttributeValue> attributeValue) {
		this.attributeValues = attributeValue;
	}

	public BigDecimal getEntropy() {
		return entropy;
	}

	public void setEntropy(BigDecimal entropy) {
		this.entropy = entropy;
	}

	public BigDecimal getInformationGain() {
		return informationGain;
	}

	public void setInformationGain(BigDecimal informationGain) {
		this.informationGain = informationGain;
	}
	
}
