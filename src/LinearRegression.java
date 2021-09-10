import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;

public class LinearRegression {
	
	public static void main(String[] args) {
		
		int instances = 400, features = 13;
		double[][] xData = new double[instances][features];
		double[] yData = new double[instances];
		
		// Reading from the Boston Housing file
		Scanner scan = null;
		try {
			FileReader reader = new FileReader("data" + System.getProperty("file.separator") + "LinearRegression" + System.getProperty("file.separator") + "BostonHousing");
			scan = new Scanner(reader);
			for (int i = 0; i < instances; i++) {
				for (int j = 0; j <= features; j++) {
					if (j == features) yData[i] = scan.nextDouble();
					else xData[i][j] = scan.nextDouble();
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			scan.close();
		}

		
		// Normalize xData
		for (int i = 0; i < features; i++) {
			double maxInput = xData[0][i], minInput = xData[0][i];
			for (int j = 1; j < instances; j++) {
				if (xData[j][i] > maxInput) maxInput = xData[j][i];
				else if (xData[j][i] < minInput) minInput = xData[j][i];
			}
			for (int j = 0; j < instances; j++) {
				xData[j][i] = (xData[j][i] - minInput)/(maxInput - minInput);
			}
		}

		// Normalize yData
		double maxInput = yData[0], minInput = yData[0];
		for (int j = 1; j < instances; j++) {
			if (yData[j] > maxInput) maxInput = yData[j];
			else if (yData[j] < minInput) minInput = yData[j];
		}
		for (int j = 0; j < instances; j++) {
			yData[j] = (yData[j] - minInput)/(maxInput - minInput);
		}

		// Split data into training, testing
		double percentTraining = 0.8; // % of data in the training set
		int lastTraining = (int) (instances*percentTraining);
		double xTraining[][] = new double[lastTraining][features];
		double yTraining[] = new double[lastTraining];
		double xTest[][] = new double[instances - lastTraining][features];
		double yTest[] = new double[instances - lastTraining];
		for (int i = 0; i < instances; i++) {
			if (i < lastTraining) {
				for (int j = 0; j < features; j++) {
					xTraining[i][j] = xData[i][j];
				}
				yTraining[i] = yData[i];
			}
			else {
				for (int j = 0; j < features; j++) {
					xTest[i-lastTraining][j] = xData[i][j];
				}
				yTest[i-lastTraining] = yData[i];
			}
		}

		
		// Create Model w/ training data & learning rate (alpha)
		LinearRegressionModel m = new LinearRegressionModel(xTraining, yTraining, 0.001);

		// Iterate batch gradient descent
		m.gradientDescent(100);

		
		// Results after training
		System.out.println("Training Set (After Training): " + m.getRMSE(m.getPrediction(xTraining), yTraining));
		System.out.println("Test Set (After Training): " + m.getRMSE(m.getPrediction(xTest), yTest));


		
	}

}
