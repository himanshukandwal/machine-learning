package dev.research.himanshu.ml.playground.decisiontree;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

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
		train(location, false);
	}
	
	public void train(String location, boolean withVariance) throws MLException {
		
		// load the data set, process it !
		Instances trainingInstances = Utility.loadInstancesFromData(Utility.loadFile(location));
		setTrainingInstances(trainingInstances);

		Map<Integer, AttributeValue> headerAttributeValues = trainingInstances.getIndexer()
				.getAttributeInstanceIndexesByName(Instances.CLASS_NAME).getAttributeValues();
		
		setHeaderEntropyValue(trainingInstances.calculateHeaderEntropy(headerAttributeValues, withVariance));
	}
	
	public void generateInitialDecisionTree() throws MLException {
		generateDecisionTree(false);
	}
	
	public void generateInitialDecisionTree(boolean withVariance) throws MLException {
		BigDecimal minInitialAttributeEntropy = null;
		Attribute bestInitialAttribute = null;
		
		for (String attributeName : getTrainingInstances().getHeader().keySet()) {
			if (!attributeName.equals(Instances.CLASS_NAME)) {

				Map<Integer, AttributeValue> attributeValues = getTrainingInstances().getIndexer()
						.getAttributeInstanceIndexesByName(attributeName).getAttributeValues();

				BigDecimal attributeEntropyValue = getTrainingInstances().calculateEntropy(attributeValues, withVariance);

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
		recursivelyGenerateDecisionTree(parent, false);
	}
	
	public void recursivelyGenerateDecisionTree(Node parent, boolean withVariance) throws MLException {
		
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

					BigDecimal attributeEntropyValue = getTrainingInstances().calculateEntropy(attributeValues, withVariance);

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
					if (attributeValue.getEntropy().compareTo(new BigDecimal(0)) != 0 
							&& currentNode.getProcessedAttributes().size() < getTrainingInstances().getHeader().size()) {
						currentNode.getCandidatesNodesList().add(attributeValue);
						recursivelyGenerateDecisionTree(currentNode);
					} else {
						DecisionNode decisionNode = new DecisionNode(currentNode);
						
						int maxCount = 0;
						int maxCountValue = 0;
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
		generateDecisionTree(false);
	}
	
	public void generateDecisionTree(boolean withVariance) throws MLException {
		try {
			generateInitialDecisionTree(withVariance);
			
			recursivelyGenerateDecisionTree(getRootNode(), withVariance);
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
		
		return new BigDecimal(trueCount).divide(new BigDecimal(instances.getInstances().length), Instances.globalMathContext)
				.multiply(new BigDecimal(100), Instances.globalMathContext);
	}
	
	public static void removeRandomNode(Node node) {
		Random random = new Random();
		Node replacingNode = null;
		Node traversingNode = node;
		int depth = 1;
		boolean isDetected = false;
		
		while (traversingNode.getChildren().size() > 0) {
			int randomAttributeIndex = random.nextInt(traversingNode.getChildren().size());
			
			if (traversingNode.getChildren().size() == 0) {
				break;
			} else {
				traversingNode = traversingNode.getChildren().get(randomAttributeIndex);
			}
			
			if (depth == (random.nextInt(10) + 3) && !isDetected) {
				replacingNode = traversingNode;
				isDetected = true;
			}
			
			depth ++;
		}
		
		DecisionNode decisionNode = traversingNode.getDecisionNode();
		
		if (replacingNode == null || replacingNode == node || replacingNode.getParent() == null)
			return;
		
		Node replacingNodeParent = replacingNode.getParent();
		replacingNodeParent.getChildren().clear();
		replacingNodeParent.setDecisionNode(decisionNode);
	}
	
	public static DecisionTreeAlgorithm pruneDecisionTree(int lvalue, int kvalue, String trainingSetLocation, String validationSetLocation, boolean withVariance) {
		DecisionTreeAlgorithm bestDta = null;
		
		try {
			DecisionTreeAlgorithm dtaWithoutVariance = new DecisionTreeAlgorithm();
			dtaWithoutVariance.train(trainingSetLocation, withVariance);
			dtaWithoutVariance.generateDecisionTree(withVariance);
			
			List<String> validationDataLines = Utility.loadFile(validationSetLocation);
			Instances validationInstances = Utility.loadInstancesFromData(validationDataLines);
			
			BigDecimal originalAccuracy = dtaWithoutVariance.validateInstances(validationInstances);
			
			for (int lvalueIndex = 0; lvalueIndex < lvalue; lvalueIndex++) {
				int mvalue = new Random().nextInt(kvalue) + 1;
				
				dtaWithoutVariance = new DecisionTreeAlgorithm();
				dtaWithoutVariance.train(trainingSetLocation, withVariance);
				dtaWithoutVariance.generateDecisionTree(withVariance);
				
				for (int mvalueIndex = 0; mvalueIndex < mvalue; mvalueIndex++)
					removeRandomNode(dtaWithoutVariance.getRootNode());
				
				BigDecimal runAccuracy = dtaWithoutVariance.validateInstances(validationInstances);
				if (originalAccuracy.compareTo(runAccuracy) < 0) {
					bestDta = dtaWithoutVariance;
					originalAccuracy = runAccuracy;
				}
			}
		} catch (MLException e) {
			e.printStackTrace();
		}
		
		return bestDta;
	}
	
	public static void main(String[] args) {
		if (args == null || args.length < 6) {
			System.out.println(" Insufficient number of parameters : ");
			System.out.println(" Please provide params : ");
			System.out.println(" a) L");
			System.out.println(" b) K");
			System.out.println(" c) <training-set>");
			System.out.println(" d) <validation-set>");
			System.out.println(" e) <test-set>");
			System.out.println(" f) <to-print>");
			
			System.exit(1);
		} 
		
		try {
			int lvalue = Integer.valueOf(args[0]);
			int kvalue = Integer.valueOf(args[1]);
			
			String trainingSetLocation = args[2];
			String validationSetLocation = args[3];
			String testSetLocation = args[4];
			Boolean toPrint = Boolean.valueOf((args[5].equalsIgnoreCase("yes") ? true : false));
			
			DecisionTreeAlgorithm dtaWithoutVariance = new DecisionTreeAlgorithm();
			DecisionTreeAlgorithm dtaWithVariance = new DecisionTreeAlgorithm();
			
			dtaWithoutVariance.train(trainingSetLocation);
			dtaWithoutVariance.generateDecisionTree();
			dtaWithVariance.train(trainingSetLocation, true);
			dtaWithVariance.generateDecisionTree(true);
			
			if (toPrint) {
				System.out.println(" Decision Tree without Variance : ");
				dtaWithoutVariance.printDecisionTree();
				
				System.out.println(" Decision Tree with Variance : ");
				dtaWithVariance.printDecisionTree();
			}
			
			List<String> validationDataLines = Utility.loadFile(validationSetLocation);
			Instances validationInstances = Utility.loadInstancesFromData(validationDataLines);
			
			System.out.println(" Validation data accuracy [without variance] : " + dtaWithoutVariance.validateInstances(validationInstances));
			System.out.println(" Validation data accuracy [with variance] : " + dtaWithVariance.validateInstances(validationInstances));
			
			List<String> testDataLines = Utility.loadFile(testSetLocation);
			Instances testInstances = Utility.loadInstancesFromData(testDataLines);
			
			System.out.println(" Test data accuracy [without variance] : " + dtaWithoutVariance.validateInstances(testInstances));
			System.out.println(" Test data accuracy [with variance] : " + dtaWithVariance.validateInstances(testInstances));
			
			DecisionTreeAlgorithm bestDtaWithoutVariance = pruneDecisionTree(lvalue, kvalue, trainingSetLocation, validationSetLocation, true);
			DecisionTreeAlgorithm bestDtaWithVariance = pruneDecisionTree(lvalue, kvalue, trainingSetLocation, validationSetLocation, false);
			if (toPrint) {
				System.out.println(" Best Decision Tree post pruning : (without variance)");
				bestDtaWithoutVariance.printDecisionTree();
				System.out.println(" Best Decision Tree accuracy (without variance) : " + bestDtaWithoutVariance.validateInstances(validationInstances));
				
				System.out.println(" Best Decision Tree post pruning : (with variance)");
				bestDtaWithVariance.printDecisionTree();
				System.out.println(" Best Decision Tree accuracy (with variance) : " + bestDtaWithVariance.validateInstances(validationInstances));
			}
			
		} catch (MLException e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
	
}