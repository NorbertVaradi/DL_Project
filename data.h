#pragma once

// ********** Image files **********

// amount of training and test data to load
const int nTrain=60000; // (max 60k)
const int nTest=10000; // (max 10k)

// training and test data
unsigned char train_image[nTrain][784];
unsigned char train_label[nTrain];
unsigned char test_image[nTest][784];
unsigned char test_label[nTest];

void loadData()
{
	FILE* file=fopen("train_images.dat", "rb");
	fseek(file, 16, SEEK_SET);
	fread(train_image, 28*28, nTrain, file);
	fclose(file);

	file=fopen("train_labels.dat", "rb");
	fseek(file, 8, SEEK_SET);
	fread(train_label, 1, nTrain, file);
	fclose(file);

	file=fopen("test_images.dat", "rb");
	fseek(file, 16, SEEK_SET);
	fread(test_image, 28*28, nTest, file);
	fclose(file);

	file=fopen("test_labels.dat", "rb");
	fseek(file, 8, SEEK_SET);
	fread(test_label, 1, nTest, file);
	fclose(file);

}