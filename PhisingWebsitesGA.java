package opt.test;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.*;
import util.linalg.Vector;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.*;

// Adapted from AbaloneTest.java from the abagail library
public class PhisingWebsitesGA {
// adapt filename so that it points to "Preprocessed_dataset_abagail" (it is the same dataset than the one used by the jupyter 
//	notebook, but it is already pre processed and does not contains the column names).
	public static String filename = ".\\Preprocessed_dataset_abagail.csv";
	public static int dataset_size = 11055;
	public static int number_features = 30;
	public static double training_ratio = 0.7; // ratio of the training set
	public static int training_set_size = (int) (training_ratio * dataset_size);
	public static int testing_set_size = dataset_size - training_set_size;

	private static Instance[] instances = initializeInstances(filename);

	private static Instance[] train_set = Arrays.copyOfRange(instances, 0, training_set_size);
	private static Instance[] test_set = Arrays.copyOfRange(instances, training_set_size, dataset_size);

	// we keep the tuned hyperparameters found for the first assigmnent (we have 30
	// input layers for the 30 features,
	// we have one hidden layer with 60 hidden units and one output layer since we
	// do binary classification.
	// We keep 500 epoch for the training even if our neural network's accuracy

	private static int inputLayer = 30, hiddenLayer = 60, outputLayer = 1;
	/*
	 *Values used to test the hyperparameters
	private static int[] trainingIterations = { 100, 200, 500, 1000, 2000, 5000 };

	private static double[] population = { 200, 500, 1000, 0.10 * dataset_size, 0.15 * dataset_size, 0.20 * dataset_size,
			0.25 * dataset_size };

	private static double[] mates = { 0.02 * dataset_size, 0.04 * dataset_size };
	private static double[] mutates = { 0.02 * dataset_size, 0.04 * dataset_size };
	*/
	private static int[] trainingIterations = {500};

	private static double[] population = { 1000 };

	private static double[] mates = { 0.04 * dataset_size };
	private static double[] mutates = {
			0.04 * dataset_size };
	
	private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

	private static ErrorMeasure measure = new SumOfSquaresError();

	private static DataSet set = new DataSet(instances);

	private static BackPropagationNetwork network[] = new BackPropagationNetwork[1];
	private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[1];
	private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[1];
	private static String[] oaNames = { "GA" };
	private static String results = "";
	private static DecimalFormat df = new DecimalFormat("0.000");

	public static void main(String[] args) {
		network[0] = factory.createClassificationNetwork(new int[] { inputLayer, hiddenLayer, outputLayer });
		nnop[0] = new NeuralNetworkOptimizationProblem(set, network[0], measure);

		// oa[0] = new RandomizedHillClimbing(nnop[0]);
		for (int a = 0; a < population.length; a++) {
			int pop = (int) (population[a]);
			for (int b = 0; b < mates.length; b++) {
				int mate = (int) (mates[b]);

				for (int c = 0; c < mutates.length; c++) {
					int mutate = (int) (mutates[c]);

					for (int m = 0; m < trainingIterations.length; m++) {

						oa[0] = new StandardGeneticAlgorithm(pop, mate, mutate, nnop[0]);

						double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0,
								correctTest = 0, incorrectTest = 0;
						train(oa[0], network[0], oaNames[0]); // trainer.train();
						end = System.nanoTime();
						trainingTime = end - start;
						trainingTime /= Math.pow(10, 9);

						Instance optimalInstance = oa[0].getOptimal();
						network[0].setWeights(optimalInstance.getData());

						double predicted, actual;
						start = System.nanoTime();
						for (int j = 0; j < train_set.length; j++) {
							network[0].setInputValues(train_set[j].getData());
							network[0].run();
							String trueLabel = train_set[j].getLabel().toString();
							trueLabel = trueLabel.replace(',', '.');
							actual = Double.parseDouble(trueLabel);

							String predictedLabel = network[0].getOutputValues().toString();
							predictedLabel = predictedLabel.replace(',', '.');
							predicted = Double.parseDouble(predictedLabel);

							double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

						}
						end = System.nanoTime();
						testingTime = end - start;
						testingTime /= Math.pow(10, 9);

						results = "\nResults for " + oaNames[0] + ": \nCorrectly classified " + correct + " instances."
								+ "\nIncorrectly classified " + incorrect
								+ " instances.\nPercent correctly classified: "
								+ df.format(correct / (correct + incorrect) * 100) + "%\nTraining time: "
								+ df.format(trainingTime) + " seconds\nTesting time: " + df.format(testingTime)
								+ " seconds\n";

						// calculation of accuracy over the test set
						double predictedTest, actualTest;

						start = System.nanoTime();
						for (int j = 0; j < test_set.length; j++) {
							network[0].setInputValues(test_set[j].getData());
							network[0].run();
							String trueLabel = test_set[j].getLabel().toString();
							trueLabel = trueLabel.replace(',', '.');
							actualTest = Double.parseDouble(trueLabel);

							String predictedLabel = network[0].getOutputValues().toString();
							predictedLabel = predictedLabel.replace(',', '.');
							predictedTest = Double.parseDouble(predictedLabel);

							double trash = Math.abs(predictedTest - actualTest) < 0.5 ? correctTest++ : incorrectTest++;

						}
						end = System.nanoTime();
						testingTime = end - start;
						testingTime /= Math.pow(10, 9);

						String testingResults = "\nResults for " + oaNames[0] + ": \nCorrectly classified "
								+ correctTest + " instances." + "\nIncorrectly classified " + incorrectTest
								+ " instances.\nPercent correctly classified: "
								+ df.format(correctTest / (correctTest + incorrectTest) * 100) + "%\nTraining time: "
								+ df.format(trainingTime) + " seconds\nTesting time: " + df.format(testingTime)
								+ " seconds\n";

						System.out.println("train accuracy for" + trainingIterations[m] + "iterations" + results);
						System.out.println("test accuracy for" + trainingIterations[m] + "iterations" + testingResults);

					}

				}

			}
		}
	}

	private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
		System.out.println("\nError results for " + oaName + "\n---------------------------");
		for (int k = 0; k < trainingIterations.length; k++) {
			int nbTrainingIterations = trainingIterations[k];
			for (int i = 0; i < nbTrainingIterations; i++) {
				oa.train();
				double error = 0;
				for (int j = 0; j < instances.length; j++) {
					network.setInputValues(instances[j].getData());
					network.run();

					Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
					String valueStr = network.getOutputValues().toString();
					valueStr = valueStr.replace(',', '.');
					example.setLabel(new Instance(Double.parseDouble(valueStr)));
					error += measure.value(output, example);
				}

				// System.out.println(df.format(error));
			}
		}
	}

	private static void shuffleArray(double[][][] lines) {
		int index;
		double[][] temp;
		Random random = new Random();
		for (int i = lines.length - 1; i > 0; i--) {
			index = random.nextInt(i + 1);
			temp = lines[index];
			lines[index] = lines[i];
			lines[i] = temp;
		}
	}

	private static Instance[] initializeInstances(String filename) {
		double[][][] lines = new double[dataset_size][][];

		try {

			// let's read of file

			BufferedReader br = new BufferedReader(new FileReader(
					new File("C:\\Users\\Gwénolé\\Desktop\\Georgia Tech\\CS 7641\\Assigment 2\\dataset.csv")));

			for (int i = 0; i < lines.length; i++) {
				Scanner scan = new Scanner(br.readLine());
				scan.useDelimiter(",");
				lines[i] = new double[2][]; // we will split each line of our dataset into the features and the
											// desired output
				lines[i][0] = new double[number_features]; // features of X
				lines[i][1] = new double[1]; // output y

				for (int j = 0; j < lines[i][0].length; j++) {
					lines[i][0][j] = Double.parseDouble(scan.next()); // we get the value of each feature for the data
																		// X
				}
				lines[i][1][0] = Double.parseDouble(scan.next()); // The output must be the last column of our dataset
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		Instance[] instances = new Instance[dataset_size];
		shuffleArray(lines); // we shuffle the lines here to get random training and testing sets later
		for (int i = 0; i < lines.length; i++) {
			double[] X = lines[i][0]; // our features
			double label = lines[i][1][0]; // our label
			instances[i] = new Instance(lines[i][0]);
			// classifications range from 0 to 30; split into 0 - 14 and 15 - 30
			instances[i].setLabel(new Instance(lines[i][1][0]));
		}
		return instances;
	}

}
