#include "grid_search.hpp"
#include "parse_args.hpp"

void parse_args(int argc, char *argv[], 
				std::string &patternsfile, 
				std::string &output,
				int &K,
				int &max_iter,
				int &n_population,
				float &elit_rate,
				float &mutation_rate
				unsigned int &seed
				)
{
	ArgumentParser argparse("Training GA BeliefPropagation 1.0");
	argparse.add_argument<std::string>("fArg", "f", "file", "Pattern Filename (with extension)", true, "");
	argparse.add_argument<std::string>("oArg", "o", "output", "Output Filename (with extension)", true, "");
	argparse.add_argument<int>("kArg", "k", "hidden", "Number of folds", false, 10);
	argparse.add_argument<int>("rArg", "r", "seed", "Seed random generator", false, 135);
	argparse.add_argument<int>("iArg", "i", "iteration", "Max Number of Iterations", false, 1000);
	argparse.add_argument<int>("nArg", "n", "population", "Size of genetic population", false, 1000);
	argparse.add_argument<float>("eArg", "e", "elite", "Percentage of population to conserve", false, .3f);
	argparse.add_argument<float>("mArg", "m", "mutation", "Probability of mutation", false, .5f);

	argparse.parse_args(argc, argv);

	argparse.get<std::string>("fArg", patternsfile);
	if(!os::file_exists(patternsfile)){std::cerr << "Pattern file not found. Given : " << patternsfile << std::endl;}
	argparse.get<std::string>("oArg", output);
	argparse.get<int>("kArg", K);
	argparse.get<int>("rArg", seed);
	argparse.get<int>("iArg", max_iter);
	argparse.get<int>("nArg", n_population);
	argparse.get<T>("eArg", elit_rate);
	argparse.get<T>("mArg", mutation_rate);
	
	return;
}

int main(int argc, char **argv)
{
	int	K, // number of folds
		n_population,// number of dna in each generation
		max_iter,// max number of iteration in the GA
		nth; // number of thread
	unsigned int seed; // random seed
	float	elit_rate, // percentage of population to conserve
			mutation_rate; // probability of mutation
	std::string patternsfile, // input filename
				output, // output filename
				genetic_out = "log/GA_" + patternsfile.substr(0, patternsfile.find_first_of(".") - 1) +
							  "_n" + std::to_string(n_population) +
							  "_i" + std::to_string(max_iter) +
							  "_e" + std::to_string(elit_rate) +
							  "_m" + std::to_string(mutation_rate) +
							  "_s" + std::to_string(seed) +
							  ".log"; // log file of parameters

	parse_args(argc, argv, patternsfile, output, K, max_iter, n_population, elit_rate, mutation_rate, seed);

#pragma omp parallel
	{
		nth = omp_get_num_threads();
	}

	Patterns<float> data(patternsfile);
	auto best_params = grid_search::genetic<ReplicatedFBP<double>>(	data, 
																	n_population,
																	max_iter,
																	elit_rate,
																	mutation_rate,
																	K,
																	seed,
																	nth, // number of thread to use for parallel GA
																	false,
																	genetic_out);

	std::ostream os(output, std::ios_base::app);
	os << patternsfile.substr(0, patternsfile.find_first_of(".") - 1) << "\t" << best_params << std::endl;
	os.close();
	
	return 0;
}

