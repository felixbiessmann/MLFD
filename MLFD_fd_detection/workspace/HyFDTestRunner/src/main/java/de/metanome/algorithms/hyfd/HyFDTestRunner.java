package de.metanome.algorithms.hyfd;

import de.metanome.algorithms.hyfd.config.Config;
import de.metanome.algorithms.hyfd.mocks.MetanomeMock;
import de.uni_potsdam.hpi.utils.CollectionUtils;

public class HyFDTestRunner {

	public void run() {
		Config conf = new Config();
		MetanomeMock.executeHyFD(conf);
	}

	public void run(String[] args) {
//		if (args.length != 4)
			this.wrongArguments(args);
		
/*		Config.Algorithm algorithm = null;
		String algorithmArg = args[0].toLowerCase();
		for (Config.Algorithm possibleAlgorithm : Config.Algorithm.values())
			if (possibleAlgorithm.name().toLowerCase().equals(algorithmArg))
				algorithm = possibleAlgorithm;
		
		Config.Database database = null;
		String databaseArg = args[1].toLowerCase();
		for (Config.Database possibleDatabase : Config.Database.values())
			if (possibleDatabase.name().toLowerCase().equals(databaseArg))
				database = possibleDatabase;
		
		Config.Dataset dataset = null;
		String datasetArg = args[2].toLowerCase();
		for (Config.Dataset possibleDataset : Config.Dataset.values())
			if (possibleDataset.name().toLowerCase().equals(datasetArg))
				dataset = possibleDataset;

		int inputTableLimit = Integer.valueOf(args[3]).intValue();
		
		if ((algorithm == null) || (database == null) || (dataset == null))
			this.wrongArguments(args);
		
		Config conf = new Config(algorithm, database, dataset, inputTableLimit, -1);
		
		this.run(conf, CollectionUtils.concat(args, "_"));
*/	}

	private void wrongArguments(String[] args) {
		StringBuilder message = new StringBuilder();
		message.append("\r\nArguments not supported: " + CollectionUtils.concat(args, " "));
		// TODO: message.append("\r\nProvide correct values: <algorithm> <database> <dataset> <inputTableLimit>");
		throw new RuntimeException(message.toString());
	}
	
}
