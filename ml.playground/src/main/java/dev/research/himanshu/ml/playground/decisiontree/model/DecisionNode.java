package dev.research.himanshu.ml.playground.decisiontree.model;

public class DecisionNode {
	private Node parent;
	private Integer decision;
	
	public DecisionNode() {}

	public DecisionNode(Node parent) {
		this();
		this.parent = parent;
	}

	public DecisionNode(Node parent, Integer decision) {
		this(parent);
		this.decision = decision;
	}

	public Node getParent() {
		return parent;
	}
	
	public void setParent(Node parent) {
		this.parent = parent;
	}
	
	public Integer getDecision() {
		return decision;
	}
	
	public void setDecision(Integer decision) {
		this.decision = decision;
	}
	
}
