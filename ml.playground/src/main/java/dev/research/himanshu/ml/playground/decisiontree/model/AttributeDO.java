package dev.research.himanshu.ml.playground.decisiontree.model;

/**
 * Training data instance attribute data structure.
 * 
 * @author Himanshu Kandwal
 */
public class AttributeDO {
	
	private String name;
	private Integer value;
	
	public AttributeDO() {}
	
	public AttributeDO(String name, Integer value) {
		super();
		this.name = name;
		this.value = value;
	}


	public String getName() {
		return name;
	}
	
	public void setName(String name) {
		this.name = name;
	}
	
	public Integer getValue() {
		return value;
	}
	
	public void setValue(Integer value) {
		this.value = value;
	}
	
	@Override
	public String toString() {
		return "(" + name + " : " + value + ")";
	}
	
}
