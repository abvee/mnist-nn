#include <stdio.h>
#include <math.h>

int token();
void load_initial();
float sigmoid(float x);

enum {IP_NEURONS = 784, HL_NEURONS = 16, OP_NEURONS = 10};


FILE *fp = NULL;
float initial_layer[784] = {0.0};
const float ALPHA = 0.01;

typedef struct {
	float value;
	float weights[IP_NEURONS];
} hl1_node;

typedef struct {
	float value;
	float weights[HL_NEURONS];
} hl2_node;

typedef struct {
	float value;
	float weights[HL_NEURONS];
} op_node;

hl1_node hl1[HL_NEURONS] = {0};
hl2_node hl2[HL_NEURONS] = {0};
op_node opl[OP_NEURONS] = {0};

int main() {
	fp = fopen("mnist/mnist_train.csv", "r");
	// skip the header line
	while (getc(fp) != '\n');

	epoch:
	int label = token();
	load_initial();

	// initialize weights to 1
	for (int i = 0; i < HL_NEURONS; i++) {
		for (int j = 0; j < IP_NEURONS; j++) {
			hl1[i].weights[j] = 1.0;
		}


		for (int j = 0; j < HL_NEURONS; j++)
			hl2[i].weights[j] = 1.0;
	}
	for (int i = 0; i < OP_NEURONS; i++)
		for (int j = 0; j < HL_NEURONS; j++)
			opl[i].weights[j] = 1.0;

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
	for (int i = 0; i < OP_NEURONS; i++)
		printf("%d: %f\n", i, opl[i].value);
	printf("\n");

	// Back propogation
	float op_errors[OP_NEURONS] = {0.0};
	float hl2_errors[HL_NEURONS] = {0.0};
	float hl1_errors[HL_NEURONS] = {0.0};

	// output layer
	for (int i = 0; i < OP_NEURONS; i++) {
		op_errors[i] = opl[i].value * (1 - opl[i].value) * (-opl[i].value);
		if (i == label - 1)
			op_errors[label-1] = opl[i].value * (1 - opl[i].value) * (1 - opl[i].value);

		for (int j = 0; j < HL_NEURONS; j++)
			opl[i].weights[j] += op_errors[i] * ALPHA * opl[i].value;
	}

	// Hidden layer 2
	for (int i = 0; i < HL_NEURONS; i++) {
		// calc errors
		float sum = 0;
		for (int j = 0; j < OP_NEURONS; j++)
			sum += op_errors[j] * opl[j].weights[i];

		hl2_errors[i] = sum * (1 - hl2[i].value) * hl2[i].value;

		// update weights
		for (int j = 0; j < HL_NEURONS; j++)
			hl2[i].weights[j] += hl2_errors[i] * ALPHA * hl2[i].value;
	}

	// Hiden layer 1
	for (int i = 0; i < HL_NEURONS; i++) {
		// calc errors
		float sum = 0;
		for (int j = 0; j < HL_NEURONS; j++)
			sum += hl2_errors[j] * hl2[j].weights[i];

		hl1_errors[i] = sum * (1 - hl1[i].value) * hl1[i].value;

		// update weights
		for (int j = 0; j < IP_NEURONS; j++)
			hl1[i].weights[j] += hl1_errors[i] * ALPHA * hl2[i].value;
	}
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
	while ((c = getc(fp)) != ',' && c != '\n')
		ret = 10 * ret + c - '0';

	return ret;
}

void load_initial() {
	for (int i = 0; i < IP_NEURONS; i++) {
		initial_layer[i] = (float) token() / 255.0;
	}
}
