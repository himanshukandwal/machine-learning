package dev.research.himanshu.ml.playground.decisiontree.model;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Map;

/**
 * Training data Instance data structure.
 * 
 * @author Himanshu Kandwal
 */
public class AttributeValue {
	
	private Node currentNode;
	private String attributeName;
	private Integer attributeValue;
	private Integer attributeValueCount;
	private BigDecimal entropy;
	private Map<Integer, Integer> classifiedCountMap;
	
	public AttributeValue() {}

	public AttributeValue(String attributeName, Integer attributeValue) {
		super();
		this.attributeName = attributeName;
		this.attributeValue = attributeValue;
	}
	
	public AttributeValue(String attributeName, Integer attributeValue, Integer attributeValueCount) {
		this(attributeName, attributeValue);
		this.attributeValueCount = attributeValueCount;
	}
	
	public Node getCurrentNode() {
		return currentNode;
	}
	
	public void setCurrentNode(Node currentNode) {
		this.currentNode = currentNode;
	}
	
	public String getAttributeName() {
		return attributeName;
	}
	
	public Integer getAttributeValue() {
		return attributeValue;
	}

	public Integer getAttributeValueCount() {
		return attributeValueCount;
	}

	public void setAttributeValueCount(Integer attributeValueCount) {
		this.attributeValueCount = attributeValueCount;
	}
	
	public BigDecimal getEntropy() {
		return entropy;
	}
	
	public void setEntropy(BigDecimal entropy) {
		this.entropy = entropy;
	}
	
	public AttributeValue incrementAttributeValueCount() {
		this.attributeValueCount ++;
		return this;
	}
	
	public Map<Integer, Integer> getClassifiedCountMap() {
		if (null == classifiedCountMap)
			classifiedCountMap = new HashMap<Integer, Integer>();
		
		return classifiedCountMap;
	}
	
	public AttributeValue insertOrIncrementClassifiedCountMap(Integer value) {
		Integer classifiedValueCount = null;
	
		if ((classifiedValueCount = getClassifiedCountMap().put(value, 1)) != null)
			getClassifiedCountMap().put(value, classifiedValueCount + 1);
	
		return this;
	}

	@Override
	public boolean equals(Object obj) {
		return ((AttributeValue) obj).attributeValue.equals(attributeValue);
	}
	
	@Override
	public int hashCode() {
		return 37 * (attributeValue == null ? 1 : (attributeValue == 0 ? 1 : attributeValue));
	}
	
}
