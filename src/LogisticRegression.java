import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;

public class LogisticRegression {
	
	public static void main(String[] args) {
		
		int instances = 450, features = 30;
		double[][] xData = new double[instances][features];
		int[] yData = new int[instances];
		
		// Read data from file
		Scanner scan = null;
		try {
			FileReader reader = new FileReader("data" + System.getProperty("file.separator") + "LogisticRegression" + System.getProperty("file.separator") + "WisconsinBreastCancer");
			scan = new Scanner(reader);
			for (int i = 0; i < instances; i++) {
				for (int j = 0; j <= features + 1; j++) {
					if (j == 0) scan.nextDouble();
					else if (j == 1) yData[i] = scan.nextInt();
					else xData[i][j-2] = scan.nextDouble();
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
	
		
		
		// Split data into training, testing
		double percentTraining = 0.7;
		int lastTraining = (int) (instances*percentTraining);
		double xTraining[][] = new double[lastTraining][features];
		int yTraining[] = new int[lastTraining];
		double xTest[][] = new double[instances - lastTraining][features];
		int yTest[] = new int[instances - lastTraining];
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
		
		
		// Model Training & Testing
		LogisticRegressionModel m = new LogisticRegressionModel(xTraining, yTraining, 0.001);

		m.gradientDescent(100);

		System.out.println("Training Set (After Training): " + m.getAccuracy(m.getPrediction(xTraining), yTraining));

		System.out.println("Test Set (After Training): " + m.getAccuracy(m.getPrediction(xTest), yTest));

	} 

}
