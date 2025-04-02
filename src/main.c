#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void init();
int token();
void load_initial();
float sigmoid(float x);

enum {IP_NEURONS = 784, HL_NEURONS = 32, OP_NEURONS = 10};


FILE *fp = NULL;
float initial_layer[784] = {0.0};
const float ALPHA = 0.5;

typedef struct {
	float value;
	float weights[IP_NEURONS];
} hl1_node;

typedef struct {
	float value;
	float weights[HL_NEURONS];
} hl2_node;

hl1_node hl1[HL_NEURONS] = {0};
hl2_node hl2[HL_NEURONS] = {0};
hl2_node opl[OP_NEURONS] = {0};

int main() {
	fp = fopen("mnist/mnist_train.csv", "r");
	// skip the header line
	while (getc(fp) != '\n');
	int epoch = 0;

	// initialize
	init();

	epoch:
	int label = token(); // output
	load_initial();

	// forward pass
	// HL1
	for (int i = 0; i < HL_NEURONS; i++) {
		float sum = 0;
		for (int j = 0; j < IP_NEURONS; j++)
			sum += initial_layer[j] * hl1[i].weights[j];

		hl1[i].value = sigmoid(sum);
	}

	// HL 2
	for (int i = 0; i < HL_NEURONS; i++) {
		float sum = 0;
		for (int j = 0; j < HL_NEURONS; j++)
			sum += hl1[j].value * hl2[i].weights[j];
		hl2[i].value = sigmoid(sum);
	}

	// Output layer
	for (int i = 0; i < OP_NEURONS; i++) {
		float sum = 0;
		for (int j = 0; j < HL_NEURONS; j++)
			sum += hl2[j].value * opl[i].weights[j];
		opl[i].value = sigmoid(sum);
	}

	// print after each pass
	int largest = 0;
	printf("Number: %d\n", label);
	for (int i = 0; i < OP_NEURONS; i++) {
		if (opl[i].value > opl[largest].value)
			largest = i;
		printf("%d: %f\n", i, opl[i].value);
	}
	printf("Final: %d\n\n", largest);

	// Back propogation
	float op_errors[OP_NEURONS] = {0.0};
	float hl2_errors[HL_NEURONS] = {0.0};
	float hl1_errors[HL_NEURONS] = {0.0};

	// hidden layer 2 -> output layer
	for (int i = 0; i < OP_NEURONS; i++) {

		// DELj = Oj * (1 - Oj) * (Tj - Oj)
		// Target for the output layer is all 0s, except for the label neuron, which is 1
		op_errors[i] = opl[i].value * (1 - opl[i].value) * (- opl[i].value);
		if (i == label)
			op_errors[label] = opl[i].value * (1.0 - opl[i].value) * (1.0 - opl[i].value);

		for (int j = 0; j < HL_NEURONS; j++)
			opl[i].weights[j] += ALPHA * op_errors[i] * hl2[j].value;
	}

	// hidden layer 1 -> hidden layer 2
	for (int i = 0; i < HL_NEURONS; i++) {
		// calc errors
		float sum = 0;
		for (int j = 0; j < OP_NEURONS; j++)
			sum += op_errors[j] * opl[j].weights[i];

		// DELj = Oj * (1 - Oj) * SUM(DEL k * Wkj)
		hl2_errors[i] = hl2[i].value * (1.0 - hl2[i].value) * sum;

		// update weights
		for (int j = 0; j < HL_NEURONS; j++)
			hl2[i].weights[j] += ALPHA * hl2_errors[i] * hl1[j].value;
	}

	// Hiden layer 1
	for (int i = 0; i < HL_NEURONS; i++) {
		// calc errors
		float sum = 0;
		for (int j = 0; j < HL_NEURONS; j++)
			sum += hl2_errors[j] * hl2[j].weights[i];

		hl1_errors[i] = hl1[i].value * (1 - hl1[i].value) * sum;

		// update weights
		for (int j = 0; j < IP_NEURONS; j++)
			hl1[i].weights[j] += ALPHA * hl1_errors[i] * initial_layer[j];
	}

	epoch += 1;
	if (epoch < 60000)
		goto epoch;
	return 0;
}

float sigmoid(float x) {
	return (float) (1.0 / (1.0 + exp(-x)));
}

// CSV tokenizer. Return next token
int token() {
	int ret = 0;
	char c;
	while ((c = getc(fp)) != ',' && c != '\n' && c != EOF)
		ret = 10 * ret + c - '0';
	return ret;
}

void load_initial() {
	for (int i = 0; i < IP_NEURONS; i++)
		initial_layer[i] = (float) token() / 255.0;
}

// set random weights for all the neurons
#include <time.h>
#include <stdlib.h>

void init() {
	srand(time(NULL));
	const int SPREAD = 8;

	// hidden layers 1 and 2 weights
	for (int i = 0; i < HL_NEURONS; i++) {

		for (int j = 0; j < IP_NEURONS; j++)
			hl1[i].weights[j] =
				(float) ((rand() % SPREAD) - SPREAD / 2)/(float) (SPREAD/2);

		for (int j = 0; j < HL_NEURONS; j++)
			hl2[i].weights[j] =
				(float) ((rand() % SPREAD) - SPREAD / 2)/(float) (SPREAD/2);
	}

	// Output layer weights
	for (int i = 0; i < OP_NEURONS; i++)
		for (int j = 0; j < HL_NEURONS; j++)
			opl[i].weights[j] =
				(float) ((rand() % SPREAD) - SPREAD / 2)/(float) (SPREAD/2);
}
