package dev.research.himanshu.ml.playground.decisiontree.model;

import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/**
 * Helper class for managing training data instances search structure.
 * 
 * @author Himanshu Kandwal
 */
public class InstanceIndexer {

	private Instances instances;
	private String attributeName;
	private Set<Integer> indexes;

	public InstanceIndexer(Instances instances) {
		this.instances = instances;
	}

	public InstanceIndexer(Instances instances, Set<Integer> indexes) {
		this(instances);
		this.indexes = indexes;
	}
	
	public InstanceIndexer(Instances instances, Set<Integer> indexes, String attributeName) {
		this(instances, indexes);
		this.attributeName = attributeName;
	}
	
	public String getAttributeName() {
		return attributeName;
	}
	
	public void setAttributeName(String attributeName) {
		this.attributeName = attributeName;
	}
	
	public Set<Integer> getIndexes() {
		if (null == indexes)
			indexes = new LinkedHashSet<Integer>();

		return indexes;
	}

	public int size() {
		return getIndexes().size();
	}
	
	public InstanceIndexer getAttributeInstanceIndexesByName(String attributeName) throws MLException {
		AttributeDO[] attributeDOs = instances.getAttributesByName(attributeName);
		Set<Integer> updatedIndexes = new HashSet<Integer>();
		
		for (int index = 0; index < attributeDOs.length; index++)
			updatedIndexes.add(index);

		return new InstanceIndexer(instances, updatedIndexes, attributeName);
	}
	
	public InstanceIndexer andThenChainAttributeInstanceIndexes(String attributeName, Integer attributeValue) throws MLException {
		AttributeDO[] attributeDOs = instances.getAttributesByName(attributeName);

		Set<Integer> updatedIndexes = new HashSet<Integer>();
		for (int index = 0; index < attributeDOs.length; index++) {
			if (getIndexes().contains(index)) {
				AttributeDO attributeDO = attributeDOs[index];
				if (attributeDO.getValue().equals(attributeValue))
					updatedIndexes.add(index);
			}
		}
		
		return new InstanceIndexer(instances, updatedIndexes, attributeName);
	}
	
	public InstanceIndexer andThenChainAttributeInstanceIndexes(String attributeName) throws MLException {
		return new InstanceIndexer(instances, new HashSet<Integer> (indexes), attributeName);
	}
	
	public Map<Integer, AttributeValue> getAttributeValues() throws MLException {
		return instances.getAttributeValues(this);
	}
	
}
