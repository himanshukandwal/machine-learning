package dev.research.himanshu.ml.playground.decisiontree;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import dev.research.himanshu.ml.playground.decisiontree.model.Attribute;
import dev.research.himanshu.ml.playground.decisiontree.model.AttributeDO;
import dev.research.himanshu.ml.playground.decisiontree.model.AttributeValue;
import dev.research.himanshu.ml.playground.decisiontree.model.DecisionNode;
import dev.research.himanshu.ml.playground.decisiontree.model.Instance;
import dev.research.himanshu.ml.playground.decisiontree.model.InstanceIndexer;
import dev.research.himanshu.ml.playground.decisiontree.model.Instances;
import dev.research.himanshu.ml.playground.decisiontree.model.MLException;
import dev.research.himanshu.ml.playground.decisiontree.model.Node;
import dev.research.himanshu.ml.playground.decisiontree.util.Utility;

/**
 * Main class for creating the decision tree from the provided input data
 * (training) set.
 * 
 * @author Himanshu Kandwal
 */
public class DecisionTreeAlgorithm {

	private Instances trainingInstances;
	private BigDecimal headerEntropyValue;
	private Node rootNode;
	
	public DecisionTreeAlgorithm() {
	}
	
	public Instances getTrainingInstances() {
		return trainingInstances;
	}
	
	public void setTrainingInstances(Instances trainingInstances) {
		this.trainingInstances = trainingInstances;
	}

	public BigDecimal getHeaderEntropyValue() {
		return headerEntropyValue;
	}
	
	public void setHeaderEntropyValue(BigDecimal headerEntropyValue) {
		this.headerEntropyValue = headerEntropyValue;
	}
	
	public Node getRootNode() {
		return rootNode;
	}
	
	public void setRootNode(Node rootNode) {
		this.rootNode = rootNode;
	}
	
	public void train(String location) throws MLException {
		
		// load the data set, process it !
		Instances trainingInstances = Utility.loadInstancesFromData(Utility.loadFile(location));
		setTrainingInstances(trainingInstances);

		Map<Integer, AttributeValue> headerAttributeValues = trainingInstances.getIndexer()
				.getAttributeInstanceIndexesByName(Instances.CLASS_NAME).getAttributeValues();
		
		setHeaderEntropyValue(trainingInstances.calculateHeaderEntropy(headerAttributeValues));
	}
	
	public void generateInitialDecisionTree() throws MLException {
		BigDecimal minInitialAttributeEntropy = null;
		Attribute bestInitialAttribute = null;
		
		for (String attributeName : getTrainingInstances().getHeader().keySet()) {
			if (!attributeName.equals(Instances.CLASS_NAME)) {

				Map<Integer, AttributeValue> attributeValues = getTrainingInstances().getIndexer()
						.getAttributeInstanceIndexesByName(attributeName).getAttributeValues();

				BigDecimal attributeEntropyValue = getTrainingInstances().calculateEntropy(attributeValues);

				if (minInitialAttributeEntropy == null || minInitialAttributeEntropy.compareTo(attributeEntropyValue) > 0) {
					minInitialAttributeEntropy = attributeEntropyValue;
					bestInitialAttribute = new Attribute(attributeName, attributeValues, attributeEntropyValue);
				}
			}
		}
		
		bestInitialAttribute.setInformationGain(getHeaderEntropyValue().subtract(minInitialAttributeEntropy, Instances.globalMathContext));
		Node rootNode = new Node(null, bestInitialAttribute, getTrainingInstances().getIndexer().getAttributeInstanceIndexesByName(bestInitialAttribute.getAttributeName()));
		
		setRootNode(rootNode);
		List<String> processedAttributes = new ArrayList<String>();
		processedAttributes.add(Instances.CLASS_NAME);
		processedAttributes.add(bestInitialAttribute.getAttributeName());		
		rootNode.setProcessedAttributes(processedAttributes);
		
		for (AttributeValue attributeValue : bestInitialAttribute.getAttributeValues().values()) {
			if (attributeValue.getEntropy().compareTo(new BigDecimal(0)) != 0) {
				rootNode.getCandidatesNodesList().add(attributeValue);
			}
			attributeValue.setCurrentNode(getRootNode());
		}
	}

	public void recursivelyGenerateDecisionTree(Node parent) throws MLException {
		
		for (Iterator<AttributeValue> candidateAttributeValueIterator = parent.getCandidatesNodesList().iterator(); candidateAttributeValueIterator.hasNext();) {
			AttributeValue candidateAttributeValue = candidateAttributeValueIterator.next();
			
			BigDecimal minAttributeEntropy = null;
			Attribute bestAttribute = null;
			InstanceIndexer bestInstanceIndexer = null;
			
			for (String attributeName : getTrainingInstances().getHeader().keySet()) {
				if (!parent.getProcessedAttributes().contains(attributeName)) {

					InstanceIndexer localInstanceIndexer = parent.getIndexer()
							.andThenChainAttributeInstanceIndexes(candidateAttributeValue.getAttributeName(),
									candidateAttributeValue.getAttributeValue())
							.andThenChainAttributeInstanceIndexes(attributeName);

					Map<Integer, AttributeValue> attributeValues = localInstanceIndexer.getAttributeValues();

					BigDecimal attributeEntropyValue = getTrainingInstances().calculateEntropy(attributeValues);

					if (minAttributeEntropy == null || minAttributeEntropy.compareTo(attributeEntropyValue) > 0) {
						minAttributeEntropy = attributeEntropyValue;
						bestAttribute = new Attribute(attributeName, attributeValues, attributeEntropyValue);
						bestInstanceIndexer = localInstanceIndexer;
					}
				}
			}
			
			if (bestAttribute != null) {
				bestAttribute.setInformationGain(parent.getAttribute().getInformationGain().subtract(minAttributeEntropy, Instances.globalMathContext));
				
				Node currentNode = new Node(parent, bestAttribute, bestInstanceIndexer);
				parent.addChild(currentNode);
				currentNode.setEdgeAttributeValue(candidateAttributeValue);
				
				candidateAttributeValueIterator.remove();
				currentNode.getProcessedAttributes().addAll(parent.getProcessedAttributes());
				currentNode.getProcessedAttributes().add(bestAttribute.getAttributeName());
				
				for (AttributeValue attributeValue : bestAttribute.getAttributeValues().values()) {
					if (attributeValue.getEntropy().compareTo(new BigDecimal(0)) != 0) {
						currentNode.getCandidatesNodesList().add(attributeValue);
						recursivelyGenerateDecisionTree(currentNode);
					} else {
						DecisionNode decisionNode = new DecisionNode(currentNode);
						
						int maxCount = -1;
						int maxCountValue = -1;
						for (Map.Entry<Integer, Integer> classifiedCountMapEntry : attributeValue.getClassifiedCountMap().entrySet()) {
							int value = classifiedCountMapEntry.getKey();
							int count = classifiedCountMapEntry.getValue();

							if (maxCount == -1 || maxCount < count) {
								maxCount = count;
								maxCountValue = value;
							}
						}
						decisionNode.setDecision(maxCountValue);
						currentNode.setDecisionNode(decisionNode);
					}
					attributeValue.setCurrentNode(currentNode);
				}
			}
		}
	}
	
	public void generateDecisionTree() throws MLException {
		try {
			generateInitialDecisionTree();
			System.out.println(" generated initial Decision tree : " + getRootNode().getAttribute().getAttributeName());
			
			recursivelyGenerateDecisionTree(getRootNode());
		} catch (MLException exception) {
			throw new MLException(" exception while generating decision tree !", exception);
		}
	}
	
	public void printDecisionTree() {
		StringBuffer treePrintingStringBuffer = new StringBuffer();
		recursivePrintDecisionTree(treePrintingStringBuffer, 0, getRootNode());
		
		System.out.println(treePrintingStringBuffer.toString());
	}
	
	public void recursivePrintDecisionTree(StringBuffer levelStatement, int level, Node node) {
		levelStatement.append("\n");
		StringBuffer seperator = new StringBuffer();

		for (int index = 0; index < level; index++)
			seperator.append(" | ");

		levelStatement.append(seperator.toString()).append(node.getAttribute().getAttributeName());

		Map<Integer, AttributeValue> nodeAttributeValuesMap = node.getAttribute().getAttributeValues();

		boolean reFillMarking = false;
		for (AttributeValue attributeValue : nodeAttributeValuesMap.values()) {
			Node matchingChildNode = null;

			for (Iterator<Node> nodesIterator = node.getChildren().iterator(); nodesIterator.hasNext();) {
				Node childnode = nodesIterator.next();
				if (childnode.getEdgeAttributeValue().getAttributeValue() == attributeValue.getAttributeValue())
					matchingChildNode = childnode;
			}

			if (reFillMarking)
				levelStatement.append("\n" + seperator.toString()).append(node.getAttribute().getAttributeName());
			
			levelStatement.append(" = " + attributeValue.getAttributeValue() + " : ");
			
			if (matchingChildNode == null) {
				levelStatement.append(node.getDecisionNode().getDecision());
				reFillMarking = true;
			} else {
				recursivePrintDecisionTree(levelStatement, level + 1, matchingChildNode);
				reFillMarking = true;
			}
		}
	}
	
	public boolean validateInstance(Instance instance) {
		AttributeDO[] attributeDOs = instance.getAttributeDOs();
		Integer attributeIndex = getTrainingInstances().getHeader().get(Instances.CLASS_NAME);
		
		if (validate(getRootNode(), attributeDOs) == attributeDOs[attributeIndex].getValue())
			return true;
		else
			return false;
	}
	
	public Integer validate(Node node, AttributeDO[] attributeDOs) {
		String currentAttributeName = node.getAttribute().getAttributeName();
		Integer attributeIndex = getTrainingInstances().getHeader().get(currentAttributeName);
		
		AttributeDO attributeDO = attributeDOs [attributeIndex];
		Integer attributeValue = attributeDO.getValue();
		
		Node nextLevelNode = null;
		for (Node childNode : node.getChildren()) {
			if (childNode.getEdgeAttributeValue().getAttributeValue().equals(attributeValue))
				nextLevelNode = childNode;
		}
		
		if (nextLevelNode == null) {
			return node.getDecisionNode().getDecision();
		}
		
		return validate(nextLevelNode, attributeDOs);
	}
	
	public BigDecimal validateInstances(Instances instances) {
		int trueCount = 0;
		
		for (Instance instance : instances.getInstances())
			trueCount += (validateInstance(instance) ? 1 : 0);
		
		return new BigDecimal(trueCount).divide(new BigDecimal(instances.getInstances().length), Instances.globalMathContext);
	}
	
}
