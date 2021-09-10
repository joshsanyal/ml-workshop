
public class LogisticRegressionModel {
	
	double[][] x; // input
	int[] y; // output
	double[] weights; // theta
	double learningRate; // alpha
	
	// Initializes model with training data and learningRate
	public LogisticRegressionModel(double[][] xData, int[] yData, double r) {
		learningRate = r;
		x = xData;
		y = yData;
		
		// Sets weights to random values from 0-1
		weights = new double[xData[0].length + 1];
		for (int i = 0; i <= xData[0].length; i++) {
			weights[i] = Math.random()*2-1;
		}
	}
	
	// Returns the accuracy of yPredict vs yReal
	public double getAccuracy(double[] yPredict, int[] yReal) {
		int correct = 0;
		for (int i = 0; i < yPredict.length; i++) {
			if ((yReal[i] == 1) == (yPredict[i] >= 0.5)) {
				correct++;
			}
		}
		return ((double) correct)/yPredict.length;
	}

	
	// Sigmoid Function
	public double sigmoid(double n) {
		return 1/(1 + Math.pow(Math.E, -1 * n));
	}
	
	// Returns array of predictions for inputted xData
	public double[] getPrediction(double[][] xData) {
		double[] predictions = new double[xData.length];
		for (int i = 0; i < xData.length; i++) {
			double yPredicted = 0;
			for (int j = 0; j < weights.length; j++) {
				if (j == 0) yPredicted += weights[j];
				else yPredicted += weights[j] * xData[i][j-1];
			}
			predictions[i] = sigmoid(yPredicted);
		}
		return predictions;
	}

	
	// Batch Gradient Descent for n iterations
	public void gradientDescent(int n) {
		for (int k = 0; k < n; k++) {
			double[] newWeights = new double[weights.length];
			double[] predictions = getPrediction(x);
			double cost = 0;
			
			for (int a = 0; a < newWeights.length; a++) {
				for (int i = 0; i < y.length; i++) {
					if (a == 0) 	cost += (predictions[i] - y[i]);
					else cost += (predictions[i] - y[i]) * x[i][a-1];
				}
				
				newWeights[a] = weights[a] - learningRate * cost / y.length;
			}
			
			// simultaneous update
			weights = newWeights;
		}
	}
}
