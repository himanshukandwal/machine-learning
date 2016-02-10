package dev.research.himanshu.ml.playground.decisiontree;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import dev.research.himanshu.ml.playground.decisiontree.model.Attribute;
import dev.research.himanshu.ml.playground.decisiontree.model.AttributeValue;
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
	private Set<String> processedAttributes = new HashSet<String>();
	private List<AttributeValue> candidatesNodesList;
	
	public DecisionTreeAlgorithm() {
		getProcessedAttributes().add(Instances.CLASS_NAME);
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
	
	public Set<String> getProcessedAttributes() {
		return processedAttributes;
	}
	
	public List<AttributeValue> getCandidatesNodesList() {
		if (candidatesNodesList == null) {
			candidatesNodesList = new ArrayList<AttributeValue>();
		}
		return candidatesNodesList;
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
			if (!getProcessedAttributes().contains(attributeName)) {

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
		setRootNode(new Node(null, bestInitialAttribute, getTrainingInstances().getIndexer().getAttributeInstanceIndexesByName(bestInitialAttribute.getAttributeName())));
		
		for (AttributeValue attributeValue : bestInitialAttribute.getAttributeValues().values()) {
			if (attributeValue.getEntropy().compareTo(new BigDecimal(0)) != 0)
				getCandidatesNodesList().add(attributeValue);
			
			attributeValue.setCurrentNode(getRootNode());
		}
		
		getProcessedAttributes().add(bestInitialAttribute.getAttributeName());
	}

	/* Tree construction will be breadth wise! */
	public void recursivelyGenerateDecisionTree() throws MLException {
		
		/* base condition */
		if (getProcessedAttributes().size() == getTrainingInstances().getHeader().size())
			return;
		
		BigDecimal minAttributeEntropy = null;
		Attribute bestAttribute = null;
		InstanceIndexer bestInstanceIndexer = null;
		AttributeValue processedCandidateAttributeValue = null;
		
		for (String attributeName : getTrainingInstances().getHeader().keySet()) {
			if (!getProcessedAttributes().contains(attributeName)) {
				
				for (AttributeValue candidateAttributeValue : getCandidatesNodesList()) {
					
					InstanceIndexer localInstanceIndexer = candidateAttributeValue.getCurrentNode().getIndexer()
							.andThenChainAttributeInstanceIndexes(candidateAttributeValue.getAttributeName(), candidateAttributeValue.getAttributeValue())
							.andThenChainAttributeInstanceIndexes(attributeName);
					
					Map<Integer, AttributeValue> attributeValues = localInstanceIndexer.getAttributeValues();
					
					BigDecimal attributeEntropyValue = getTrainingInstances().calculateEntropy(attributeValues);
					
					if (minAttributeEntropy == null || minAttributeEntropy.compareTo(attributeEntropyValue) > 0) {
						minAttributeEntropy = attributeEntropyValue;
						bestAttribute = new Attribute(attributeName, attributeValues, attributeEntropyValue);
						bestInstanceIndexer = localInstanceIndexer;
						processedCandidateAttributeValue = candidateAttributeValue;
					}
				}
			}
		}

		bestAttribute.setInformationGain(processedCandidateAttributeValue.getCurrentNode().getAttribute().getInformationGain()
				.subtract(minAttributeEntropy, Instances.globalMathContext));
		
		Node currentNode = new Node(processedCandidateAttributeValue.getCurrentNode(), bestAttribute, bestInstanceIndexer);
		processedCandidateAttributeValue.getCurrentNode().addChild(currentNode);
		currentNode.setEdgeAttributeValue(processedCandidateAttributeValue);
		
		getCandidatesNodesList().remove(processedCandidateAttributeValue);
		
		for (AttributeValue attributeValue : bestAttribute.getAttributeValues().values()) {
			if (attributeValue.getEntropy().compareTo(new BigDecimal(0)) != 0) {
				getCandidatesNodesList().add(attributeValue);
				attributeValue.setCurrentNode(currentNode);
			}
		}
		
		System.out.println(" processed attribute : " + bestAttribute.getAttributeName());
		getProcessedAttributes().add(bestAttribute.getAttributeName());
	}
	
	public void generateDecisionTree() throws MLException {
		try {
			generateInitialDecisionTree();
			System.out.println(" generated initial Decision tree : " + getRootNode().getAttribute().getAttributeName());
			
			while (getProcessedAttributes().size() != getTrainingInstances().getHeader().size() && getCandidatesNodesList().size() > 0) {
				recursivelyGenerateDecisionTree();
			}
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
			seperator.append("| ");

		levelStatement.append(seperator.toString()).append(node.getAttribute().getAttributeName() + " [ " + node.getInformationGain() +" ] ");

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
				levelStatement.append("\n" + seperator.toString()).append(node.getAttribute().getAttributeName() + " [ " + node.getInformationGain() +" ]");
			
			levelStatement.append(" = " + attributeValue.getAttributeValue() + " (" + attributeValue.getEntropy() + ")" + " : ");
			
			if (matchingChildNode == null) {
				int maxCount = -1;
				int maxCountValue = -1;
				
				for (Map.Entry<Integer, Integer> classifiedCountMapEntry : attributeValue.getClassifiedCountMap().entrySet()) {
					int value = classifiedCountMapEntry.getKey();
					int count = classifiedCountMapEntry.getValue();

					maxCountValue = (maxCount == -1 || maxCount < count ? value : maxCountValue);
					maxCount = (maxCount == -1 || maxCount < count ? count : maxCount);
				}

				levelStatement.append(maxCountValue);
				reFillMarking = true;
			} else {
				recursivePrintDecisionTree(levelStatement, level + 1, matchingChildNode);
				reFillMarking = true;
			}
		}
	}
	
	public static void main(String[] args) throws MLException {
		DecisionTreeAlgorithm dta = new DecisionTreeAlgorithm();
		dta.train(
				"/Users/Heman/Documents/workstation/Developement_Studio/Java_Laboratory/Leisure_WorkZones/Eclipse_Workspaces/Machine-Learning/ml.playground/src/main/resources/data_sets_1/training_set.csv");
		
		System.out.println(" -- loaded training data !");
		
		dta.generateDecisionTree();
		dta.printDecisionTree();
	}
}
