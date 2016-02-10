package dev.research.himanshu.ml.playground.decisiontree.model;

/**
 * Training data Instance data structure.
 * 
 * @author Himanshu Kandwal
 */
public class Instance {

	private AttributeDO[] attributeDOs;
	
	public Instance() {}
	
	public Instance(AttributeDO[] attributeDOs) {
		super();
		this.attributeDOs = attributeDOs;
	}
	
	public AttributeDO[] getAttributeDOs() {
		return attributeDOs;
	}
	
	public void setAttributes(AttributeDO[] attributeDOs) {
		this.attributeDOs = attributeDOs;
	}
	
	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer(" { ");
		for (int index = 0; index < attributeDOs.length; index ++) {
			sb.append(attributeDOs[index]);
			
			if (index < attributeDOs.length -1)
				sb.append(", ");
		}
		return sb.append(" } ").toString();
	}
}
