package dev.research.himanshu.ml.playground.decisiontree.model;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

/**
 * Node Data Structure of the decision tree.
 * 
 * @author Himanshu Kandwal
 */
public class Node {
	private Node parent;
	private Attribute attribute;
	private BigDecimal informationGain;
	private List<Node> children;
	private InstanceIndexer indexer;
	private AttributeValue edgeAttributeValue;
	
	public Node() {}

	public Node(Attribute attribute, InstanceIndexer indexer) {
		this();
		this.attribute = attribute;
		this.indexer = indexer;
		this.informationGain = attribute.getInformationGain();
	}
	
	public Node(Node parent, Attribute attribute, InstanceIndexer indexer) {
		this(attribute, indexer);
		this.parent = parent;
	}
	
	public Node getParent() {
		return parent;
	}

	public void setParent(Node parent) {
		this.parent = parent;
	}

	public AttributeValue getEdgeAttributeValue() {
		return edgeAttributeValue;
	}
	
	public void setEdgeAttributeValue(AttributeValue edgeAttributeValue) {
		this.edgeAttributeValue = edgeAttributeValue;
	}
	
	public Attribute getAttribute() {
		return attribute;
	}

	public void setAttribute(Attribute attribute) {
		this.attribute = attribute;
	}

	public List<Node> getChildren() {
		if (children == null) {
			children = new ArrayList<Node>();
		}
		return children;
	}

	public void setChildren(List<Node> children) {
		this.children = children;
	}
	
	public void setInformationGain(BigDecimal informationGain) {
		this.informationGain = informationGain;
	}
	
	public BigDecimal getInformationGain() {
		return informationGain;
	}
	
	public void addChild(Node node) {
		getChildren().add(node);
	}
	
	public InstanceIndexer getIndexer() {
		return indexer;
	}
	
	public void setIndexer(InstanceIndexer indexer) {
		this.indexer = indexer;
	}

}