#include "classifier.hpp"
#include "score_coef.hpp"
#include "parse_args.hpp"

void parse_args(int argc, char *argv[], 
				std::string &trainfile, 
				std::string &testfile, 
				std::string &output,
				std::string &paramsfile
				)
{
	ArgumentParser argparse("Validation BeliefPropagation 1.0");
	argparse.add_argument<std::string>("fArg", "f", "train", "Train Filename (with extension)", true, "");
	argparse.add_argument<std::string>("tArg", "t", "test", "Test Filename (with extension)", true, "");
	argparse.add_argument<std::string>("pArg", "p", "params", "Parameters Filename (with extension)", true, "");
	argparse.add_argument<std::string>("oArg", "o", "output", "Output Filename (with extension)", true, "");

	argparse.parse_args(argc, argv);

	argparse.get<std::string>("fArg", trainfile);
	if(!os::file_exists(trainfile)){std::cerr << "Train file not found. Given : " << trainfile << std::endl;}
	argparse.get<std::string>("tArg", testfile);
	if(!os::file_exists(testfile)){std::cerr << "Train file not found. Given : " << testfile << std::endl;}
	argparse.get<std::string>("pArg", paramsfile);
	if(!os::file_exists(paramsfile)){std::cerr << "Parameters file not found. Given : " << paramsfile << std::endl;}
	argparse.get<std::string>("oArg", output);
	
	return;
}

int main(int argc, char **argv)
{
	std::string trainfile, // train filename
				testfile, // test filename
				paramsfile, // parameters filename
				output; // output filename

	// miss import hyperparams

	Patterns<double> train(trainfile);
	Patterns<double> test(testfile);

	ReplicatedFBP<double> fbp;
	fbp.train<MagP64<double>>(	train, 
								params);
	int *label_predict = fbp.test(test);
	auto mcc_fbp = score::matthews_corrcoef(test.output, label_predict, test.Nout);
	auto acc_fbp = score::accuracy(test.output, label_predict, test.Nout);

	std::ofstream os(output, std::ios_base::app);
	os << mcc_fbp << "\t" << acc_fbp << std::endl;

	return 0;
}
