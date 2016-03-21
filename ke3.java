import java.util.*;
import java.io.*;


public class ke3{

	/******************Create InputImage object*****************/
	public static class InputImage {
		public String fileName;
		public double gender;
		public int[] pixel;

		InputImage(String fname, double g, int[] p){
			this.fileName = fname;
			this.gender = g;
			this.pixel = p;
		}
	}

	/******************Constatnt*****************/
	public static int SEED = 1000;
	// public static int STD_SAMPLE = 1;	// it is tester
	public static int NUMBER_OF_TIMES = 10;	// it is epochs

	/******************Create Neuron*****************/
	public static double MALE = 0.0;
	public static double FEMALE = 1.0;
	public static double INITIAL_WEIGHT = 0.0;
	public static double WEIGHT_UPPER_BOUND = 10000;
	public static double SIGMA = 0.05;
	public static double SIGMA_LOWER_BOUND = 0.01;
	public static int PHOTO_SIZE = 128 * 120;
	public static int[] LAYER = {5, 1};	//	create 2 layers where  LAYER[0] is hidden layer with 5 hidden nodes; LAYER[1] is output layer with 1 output node
	public static Neuron[][] NEURONS = new Neuron[LAYER.length][5];
	// public static double SCALE_WEIGHT = 30;

	// Neuron class
	public static class Neuron {
		double[] weight; // Incoming weights
		Neuron[] parent; // Previous layer of neurons -- remark: null at first
		Neuron[] child; // Next layer of Neurons -- remark: null for last
		int layer; // Index of layer
		int index; // Index of current node in this layer
		double[] input; // Inputs
		double result_weightInput = 0; // Weighted Input
		double output = 0; // Output
		double delta = 0; // Delta

		//	Neuron constructor
		Neuron(int l, int index, int num, int p, int c) {
			this.layer = l;
			this.index = index;

			this.weight = new double[num];
			this.input = new double[num];
			this.parent = new Neuron[p];
			this.child = new Neuron[c];


			// Initialize weights and inputs
			for(int i = 0; i < num; i++){
				this.weight[i] = (Math.random() * ((0.5 - (-0.5))) + (-0.5));
				this.input[i] = 0;
			}
			
		}

		public void calHiddenOut(){
			for(int i = 0; i < PHOTO_SIZE; i++){
				this.result_weightInput += this.weight[i] * this.input[i];
			}
			this.output =  formula(this.result_weightInput);
		}

		public void feed(int j){
			
			//inputs of output node = output of hidden nodes
			this.child[0].input[j] = this.output;
			
		}

		public void calOutputOut(){
			for(int i = 0; i < 5; i++){
				this.result_weightInput += this.weight[i] * this.input[i];
			}
			this.output =  formula(this.result_weightInput);
		}

		public double formula(double input){
			return sigmoidFcn(input);
		}

		public void calOutputError(double target_gender){
			this.delta = this.output * (1 - this.output) * (target_gender - this.output);
		}

		public void calHiddenError(int i){
			// this.child[0] = output node
			this.delta =  this.output * (1 - this.output) * (this.child[0].weight[i] * this.child[0].delta);
		}

		public void adjust(){

			double delta_weight = 0;

			// System.out.println("[" + this.layer + "-" + this.index + "]" + ", result_weightInput = " + this.result_weightInput + ", output = " + this.output + ", Delta = " + this.delta + ", derivative = " + d + ", Factor = " + f);

			for(int i = 0; i < this.weight.length; i++){
				delta_weight = SIGMA * this.delta * this.input[i];
				this.weight[i] = this.weight[i] + delta_weight;

			}
		}


	}

	/******************derivative Function***************/
	public static double find_derivative(double x){
		double y = sigmoidFcn(x);
		return y * (1 - y);
	}


	/******************Sigmoid Function***************/
	public static double sigmoidFcn(double input){
		double output = 1.0 / (1.0 + Math.exp(-input));
		// System.out.println(output);
		if(output < 0.1)
			return 0.1;
		else if(output > 0.9)
			return 0.9;
		else
			return output;
	}

	/******************Create ANN********************/
	public static void createANN(boolean test){

		//hidden layer creation
		for(int index = 0; index < 5; index++){
			NEURONS[0][index] = new Neuron(0, index, PHOTO_SIZE, 0, 1);
		}

		//output layer creation
		NEURONS[1][0] = new Neuron(1, 0, 5, 5, 0);

		//connect hidden layer child to output node
		for(int index = 0; index < 5; index++){
			NEURONS[0][index].child[0] = NEURONS[1][0];
		}

		//connect output layer parent to hidden nodes
		for(int index = 0; index < 5; index++){
			NEURONS[1][0].parent[index] = NEURONS[0][index];
		}

		// put weights into network from weight.txt
		if(test == true){
			int counter = 1;
			int position = 0;
			double w;
			try{
				Scanner s = new Scanner("./weight.txt");

				while(s.hasNextDouble()){

					w = s.nextDouble();

					if(position == 15361 || position == 30721 || position == 46081 || position == 61441 || position == 76801)
						position = 0;

					if(counter <= 15360)
						NEURONS[0][0].weight[position] = w;
					else if(counter > 15360 && counter <= 30720)
						NEURONS[0][1].weight[position] = w;
					else if(counter > 30720 && counter <= 46080)
						NEURONS[0][2].weight[position] = w;
					else if(counter > 46080 && counter <= 61440)
						NEURONS[0][3].weight[position] = w;
					else if(counter > 61440 && counter <= 76800)
						NEURONS[0][4].weight[position] = w;
					else
						NEURONS[1][0].weight[position] = w;

					position++;
				}

			}
			catch(Exception e){
				e.printStackTrace();
			}
		}

	}





	/******************Train process*****************/
	public static void fiveFoldValidation(LinkedList<InputImage> train){
		for(int i = 1; i <= NUMBER_OF_TIMES; i++ ){
			double mean = 0;
			double sd = 0;
			double sum = 0;
			double[] iteration_result = new double[5];


			// shuffle training set
			
			// Collections.shuffle(train, new Random(SEED));

			// create 5 blocks
			LinkedList<InputImage> block1 = new LinkedList<InputImage>();
			LinkedList<InputImage> block2 = new LinkedList<InputImage>();
			LinkedList<InputImage> block3 = new LinkedList<InputImage>();
			LinkedList<InputImage> block4 = new LinkedList<InputImage>();
			LinkedList<InputImage> block5 = new LinkedList<InputImage>();

			for( int index = 0; index < 273; index++){
				if(index < 54){
					block1.add(train.get(index));
				}
				else if(index >= 54 && index < 108){
					block2.add(train.get(index));
				}
				else if(index >= 108 && index < 162){
					block3.add(train.get(index));
				}
				else if(index >= 162 && index < 216){
					block4.add(train.get(index));
				}
				else{
					block5.add(train.get(index));
				}
			}

			// store the blocks into a list
			LinkedList<LinkedList<InputImage>> blockList = new LinkedList<LinkedList<InputImage>>();
			blockList.add(block1);
			blockList.add(block2);
			blockList.add(block3);
			blockList.add(block4);
			blockList.add(block5);

			// start training
			int testBlockIndex;
			for(testBlockIndex = 0; testBlockIndex < 5; testBlockIndex++){


				// set up 4 counters
				double correct_male = 0;
				double correct_female = 0;
				double total_male = 0;
				double total_female = 0;

				// train before testing
				for(int blockIndex = 0; blockIndex < 5; blockIndex++){
					
					// when blcokIndex equals to testBlockIndex, skip the block. We don't use this block for training
					if(testBlockIndex != blockIndex){
						
						
						LinkedList<InputImage> currentBlock = blockList.get(blockIndex);


						
						for(int j = 0; j < currentBlock.size(); j++){
							
							InputImage temp_image = new InputImage(currentBlock.get(j).fileName, currentBlock.get(j).gender, currentBlock.get(j).pixel);

							double predict =  train(temp_image.gender, temp_image.pixel);
							if(temp_image.gender == MALE)
								total_male++;
							if(temp_image.gender == FEMALE)
								total_female++;
							if(temp_image.gender == MALE && predict <= 0.5)
								correct_male++;
							if(temp_image.gender == FEMALE && predict > 0.5)
								correct_female++;
							// System.out.println("\n\n" + temp_image.fileName + "\t" + temp_image.gender + "\t ....predict = " + predict);
							

							
							
							
							
						}
					}
					// else{
						
					// 	System.out.println("Experiment " + i + ", testBlock " + blockIndex + " is selected in this iteration. Skip!\n");
					// }
				}

				// /********************iteration result*********************/
				// System.out.print("experiment " + i +", iteration " + testBlockIndex + ":\n" + 
				// 						 	 "Males(" + (int)correct_male + "/" + (int)total_male + " = " + (int)((correct_male / total_male) * 100) + "%)\n" +
				// 				 		 	 "Females(" + (int)correct_female + "/" + (int)total_female + " = " + (int)((correct_female / total_female) * 100) + "%)\n" +
				// 				 		 	 "Total(" + (int)(correct_female + correct_male) + "/" + (int)(total_female + total_male) + " = " + (int)(((correct_female + correct_male)/ (total_female + total_male)) * 100) + "%)\n\n");

				

				// mean +=  (int)(((correct_female + correct_male)/ (total_female + total_male)) * 100);
				// iteration_result[testBlockIndex] = (int)(((correct_female + correct_male)/ (total_female + total_male)) * 100);


				// after training, we use the test block to test
				double[] test_result = testMethod(blockList.get(testBlockIndex), false);

				// total_male += test_result[0];
				// total_female += test_result[1];
				// correct_male += test_result[2];
				// correct_female += test_result[3];


				mean +=  (int)(((correct_female + correct_male)/ (total_female + total_male)) * 100);
				iteration_result[testBlockIndex] = (int)(((correct_female + correct_male)/ (total_female + total_male)) * 100);

				/********************iteration result*********************/
				System.out.print("experiment " + i +", iteration " + testBlockIndex + ":\n" + 
										 	 "Males(" + (int)correct_male + "/" + (int)total_male + " = " + (int)((correct_male / total_male) * 100) + "%)\n" +
								 		 	 "Females(" + (int)correct_female + "/" + (int)total_female + " = " + (int)((correct_female / total_female) * 100) + "%)\n" +
								 		 	 "Total(" + (int)(correct_female + correct_male) + "/" + (int)(total_female + total_male) + " = " + (int)(((correct_female + correct_male)/ (total_female + total_male)) * 100) + "%)\n\n");
				

				// reset hidden layer nodes
				for(int s = 0; s < PHOTO_SIZE; s++){
					NEURONS[0][0].weight[s] = (Math.random() * ((0.5 - (-0.5))) + (-0.5));
					NEURONS[0][1].weight[s] = (Math.random() * ((0.5 - (-0.5))) + (-0.5));
					NEURONS[0][2].weight[s] = (Math.random() * ((0.5 - (-0.5))) + (-0.5));
					NEURONS[0][3].weight[s] = (Math.random() * ((0.5 - (-0.5))) + (-0.5));
					NEURONS[0][4].weight[s] = (Math.random() * ((0.5 - (-0.5))) + (-0.5));
				}

				// reset output layer node
				for(int s = 0; s < 5; s++)
					NEURONS[1][0].weight[s] = (Math.random() * ((0.5 - (-0.5))) + (-0.5));

				// reset result_weightInput
				NEURONS[0][0].result_weightInput = 0;
				NEURONS[0][1].result_weightInput = 0;
				NEURONS[0][2].result_weightInput = 0;
				NEURONS[0][3].result_weightInput = 0;
				NEURONS[0][4].result_weightInput = 0;
				NEURONS[1][0].result_weightInput = 0;

				// reset delta
				NEURONS[0][0].delta = 0;
				NEURONS[0][1].delta = 0;
				NEURONS[0][2].delta = 0;
				NEURONS[0][3].delta = 0;
				NEURONS[0][4].delta = 0;
				NEURONS[1][0].delta = 0;


			}
			/********************experimental result*********************/
			// calculate mean
			mean = mean / 5;

			// calculate SD
			for (int x = 0; x < 5; x++) {
				sum += Math.pow(iteration_result[x] - mean,2);
			}	

			sd = Math.sqrt(sum/5);

			System.out.println("Experiment " + i + "\nMean of accuracy: " + mean + "%\nStandard Deviation: " + sd + "\n");
			





		}

		/********************** store weights *******************/
		storeweight();

		/********************** reset weights, result_weightInput and delta for next experiment *******************/

		// reset hidden layer nodes
		for(int s = 0; s < PHOTO_SIZE; s++){
			NEURONS[0][0].weight[s] = (Math.random() * ((0.5 - (-0.5))) + (-0.5));
			NEURONS[0][1].weight[s] = (Math.random() * ((0.5 - (-0.5))) + (-0.5));
			NEURONS[0][2].weight[s] = (Math.random() * ((0.5 - (-0.5))) + (-0.5));
			NEURONS[0][3].weight[s] = (Math.random() * ((0.5 - (-0.5))) + (-0.5));
			NEURONS[0][4].weight[s] = (Math.random() * ((0.5 - (-0.5))) + (-0.5));
		}

		// reset output layer node
		for(int s = 0; s < 5; s++)
			NEURONS[1][0].weight[s] = (Math.random() * ((0.5 - (-0.5))) + (-0.5));

			// reset result_weightInput
			NEURONS[0][0].result_weightInput = 0;
			NEURONS[0][1].result_weightInput = 0;
			NEURONS[0][2].result_weightInput = 0;
			NEURONS[0][3].result_weightInput = 0;
			NEURONS[0][4].result_weightInput = 0;
			NEURONS[1][0].result_weightInput = 0;
		

		// // reset delta
		// NEURONS[0][0].delta = 0;
		// NEURONS[0][1].delta = 0;
		// NEURONS[0][2].delta = 0;
		// NEURONS[0][3].delta = 0;
		// NEURONS[0][4].delta = 0;
		// NEURONS[1][0].delta = 0;
	}

	
	/******************Test process*****************/
	public static double[] testMethod(LinkedList<InputImage> test, boolean show){

		// set up 4 counters
		double correct_male = 0;
		double correct_female = 0;
		double total_male = 0;
		double total_female = 0;

		double predict;
		for(int i = 0; i < test.size(); i++){
			predict = test(test.get(i).pixel);

			if(predict <= 0.5 && show){
				System.out.println("Test file: " + test.get(i).fileName + " is MALE");
			}
			else if(predict > 0.5 && show){
				System.out.println("Test file: " + test.get(i).fileName + " is FEMALE");
			}

			// show == false means the program is in training mode
			if(show == false){

				if(test.get(i).gender == MALE)
					total_male++;
				if(test.get(i).gender == FEMALE)
					total_female++;
				if(test.get(i).gender == MALE && predict <= 0.5)
					correct_male++;
				if(test.get(i).gender == FEMALE && predict > 0.5)
					correct_female++;
			}
		}

		System.out.print(				 	 "Males(" + (int)correct_male + "/" + (int)total_male + " = " + (int)((correct_male / total_male) * 100) + "%)\n" +
								 		 	 "Females(" + (int)correct_female + "/" + (int)total_female + " = " + (int)((correct_female / total_female) * 100) + "%)\n" +
								 		 	 "Total(" + (int)(correct_female + correct_male) + "/" + (int)(total_female + total_male) + " = " + (int)(((correct_female + correct_male)/ (total_female + total_male)) * 100) + "%)\n\n");

		double[] result = {total_male, total_female, correct_male, correct_female};

		return result;
	}

	public static double train(double gender, int[] pixel){
		// part 0: take in input
		int i, j;
		for(i = 0; i < PHOTO_SIZE; i++){
			for(j = 0; j < LAYER[0]; j++){
				NEURONS[0][j].input[i] = pixel[i];
			}
		}

		// part 1: propagate the input foward through the network
		//calculate output for hidden layer
		for(i = 0; i < 5; i++){
			NEURONS[0][i].calHiddenOut();
		}

		// feed output from hidden layer to input of output layer
		for(i = 0; i < 5; i++){
			NEURONS[0][i].feed(i);
		}

		// calculate output for output layer
		NEURONS[1][0].calOutputOut();


		// Propagate the errors backward through network		
		// part 2 - find error in output layer
		NEURONS[1][0].calOutputError(gender);

		// part 3 - find error in hidden layer
		for(i = 0; i < 5; i++){
			NEURONS[0][i].calHiddenError(i);
		}

		// part 4 - adjust weights
		for(i = 0; i < LAYER.length; i++){
			for(j = 0; j < LAYER[i]; j++){
				NEURONS[i][j].adjust();
			}
		}

		// return output from output layer
		// System.out.println(NEURONS[LAYER.length - 1][0].output);
		return NEURONS[1][0].output;
	}


	/******************Test process*****************/
	public static double test(int[] pixel){
		// part 0: take in input
		int i, j;
		for(i = 0; i < PHOTO_SIZE; i++){
			for(j = 0; j < LAYER[0]; j++){
				NEURONS[0][j].input[i] = pixel[i];
			}
		}

		// part 1: propagate the input foward through the network
		//calculate output for hidden layer
		for(i = 0; i < 5; i++){
			NEURONS[0][i].calHiddenOut();
		}

		// feed output from hidden layer to input of output layer
		for(i = 0; i < 5; i++){
			NEURONS[0][i].feed(i);
		}

		// calculate output for output layer
		NEURONS[1][0].calOutputOut();

		// return output from output layer
		return NEURONS[1][0].output;
	}





	/************ (Main) Program start from here***********/
	// User will pass in file name, -train and -test option
	public static void main(String[] args){

		// Error message
		if (args.length <= 0) {
			System.out.println("Please choose -test or -train");
			return;
		}
		if(args.length >1){
			throw new IllegalArgumentException("unknown argument");
		}

		int k = 0;
		boolean trainBool = false;
		boolean testBool = false;
		// String fileName = "";


		while (k < args.length){
				if (args[k].equalsIgnoreCase("-train")) {
					trainBool = true;
					k++;

					// // user input file name after -train
					// fileName = args[1];
					// i++;

				} 
				else if (args[k].equalsIgnoreCase("-test")) {
					testBool = true;
					k++;

					// // user input file name after -test
					// fileName = args[1];
					// i++;
					
				} 
				else{
					
					throw new IllegalArgumentException("unknown argument");

				}
		}


		

		/******************create ANN**************************/
		createANN(testBool);

		/******************file initialization*****************/
		// create linkedlist to the training set and testing set
		LinkedList<InputImage> train = new LinkedList<InputImage>();
		LinkedList<InputImage> test = new LinkedList<InputImage>();
		if(trainBool == true){		
			

			// train male with Male
			File maleFolder = new File("./Male");
			LinkedList<String> trainWithMaleSet = generateFileList(maleFolder);
			for(int i =0; i < trainWithMaleSet.size(); i++){
				String fileName = trainWithMaleSet.get(i);
				File txtName = new File("./Male/" + fileName);
				int[] pixel = readPixel(txtName);
				train.add(new InputImage(fileName,MALE,pixel));
			}

			// train female with Female
			File femaleFolder = new File("./Female");
			LinkedList<String> trainWithFemaleSet = generateFileList(femaleFolder);
			for(int i =0; i < trainWithFemaleSet.size(); i++){
				String fileName = trainWithFemaleSet.get(i);
				File txtName = new File("./Female/" + fileName);
				int[] pixel = readPixel(txtName);
				train.add(new InputImage(fileName,FEMALE,pixel));
			}
		}

		// if user select test mode, read test sets
		if(testBool == true){

			//test male with TestMaleSet
			File testFolder = new File("./Test");
			LinkedList<String> testingSet = generateFileList(testFolder);
			for(int i =0; i < testingSet.size(); i++){
				
				String fileName = testingSet.get(i);
				File txtName = new File("./Test/" + fileName);
				int[] pixel = readPixel(txtName);
				test.add(new InputImage(fileName,0.5,pixel));
			}

		}

		/**************start traning***********/
		if(trainBool == true && testBool == false){
			// randomize training set
			Collections.shuffle(train, new Random(SEED));

			fiveFoldValidation(train);
		}

		if(testBool = true && trainBool == false){
			// System.out.println(trainBool);
			// System.out.println(testBool);
			testMethod(test, true);
		}
	 
	}




	/**************get the fileName from the given test / train set***********/
	public static LinkedList<String> generateFileList(File path){
		LinkedList<String> file = new LinkedList<String>();
		try{
			for (final File fileEntry : path.listFiles()) {
           		file.add(fileEntry.getName());
	    	}
		}
		catch(Exception e){
			e.printStackTrace();
		}
		return file;
	}

	/**************Read the pixels in this file***********/
	public static int[] readPixel(File fileName){
		int[] pixel = new int[PHOTO_SIZE];
		int i = 0;
		try{
			Scanner s = new Scanner(fileName);
			while(s.hasNextInt()){

				pixel[i] = s.nextInt();
				// System.out.println(pixel[i]);
				i++;
			}
		}
		catch(Exception e){
			e.printStackTrace();
		}
		return pixel;
	}

	public static void storeweight(){
		try{
			PrintWriter writer = new PrintWriter("weight.txt");

			//store weights from input layer to hidden layer
			for(int i = 0; i < 5; i++){
				for(int j = 0; j < PHOTO_SIZE; j++){
					writer.println(NEURONS[0][i].weight[j]);
				}
			}
			
			//store weights from hidden layer to output layer
			for(int j = 0; j < 5; j++){
				writer.println(NEURONS[1][0].weight[j]);
			}

			writer.close();
		} catch(Exception e) {
    		e.printStackTrace();
		}
	}

}