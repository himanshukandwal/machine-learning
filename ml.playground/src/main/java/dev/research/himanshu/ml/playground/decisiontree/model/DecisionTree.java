package dev.research.himanshu.ml.playground.decisiontree.model;

/**
 * Decision tree data structure.
 * 
 * @author Himanshu Kandwal
 */
public class DecisionTree {
	
	private Node _root;
	
	public DecisionTree() {}
	
	public Node getRoot() {
		return _root;
	}
	
	public void setRoot(Node root) {
		this._root = root;
	}
}
