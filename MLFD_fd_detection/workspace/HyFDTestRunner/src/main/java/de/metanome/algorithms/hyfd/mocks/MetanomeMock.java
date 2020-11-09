package de.metanome.algorithms.hyfd.mocks;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import de.metanome.algorithm_integration.AlgorithmConfigurationException;
import de.metanome.algorithm_integration.AlgorithmExecutionException;
import de.metanome.algorithm_integration.ColumnIdentifier;
import de.metanome.algorithm_integration.configuration.ConfigurationSettingFileInput;
import de.metanome.algorithm_integration.input.InputGenerationException;
import de.metanome.algorithm_integration.input.RelationalInput;
import de.metanome.algorithm_integration.input.RelationalInputGenerator;
import de.metanome.algorithm_integration.results.FunctionalDependency;
import de.metanome.algorithm_integration.results.Result;
import de.metanome.algorithms.hyfd.HyFD;
import de.metanome.algorithms.hyfd.config.Config;
import de.metanome.backend.input.file.DefaultFileInputGenerator;
import de.metanome.backend.result_receiver.ResultCache;
import de.metanome.backend.result_receiver.ResultReceiver;
import de.uni_potsdam.hpi.utils.CollectionUtils;
import de.uni_potsdam.hpi.utils.FileUtils;

public class MetanomeMock {
	
	public static List<ColumnIdentifier> getAcceptedColumns(RelationalInputGenerator relationalInputGenerator) throws InputGenerationException, AlgorithmConfigurationException {
		List<ColumnIdentifier> acceptedColumns = new ArrayList<>();
		RelationalInput relationalInput = relationalInputGenerator.generateNewCopy();
		String tableName = relationalInput.relationName();
		for (String columnName : relationalInput.columnNames())
			acceptedColumns.add(new ColumnIdentifier(tableName, columnName));
		return acceptedColumns;
    }
    
	public static void executeHyFD(Config conf) {
		try {
			HyFD hyFD = new HyFD();
			
			RelationalInputGenerator relationalInputGenerator = new DefaultFileInputGenerator(new ConfigurationSettingFileInput(
					conf.inputFolderPath + conf.inputDatasetName + conf.inputFileEnding, true,
					conf.inputFileSeparator, conf.inputFileQuotechar, conf.inputFileEscape, conf.inputFileStrictQuotes, 
					conf.inputFileIgnoreLeadingWhiteSpace, conf.inputFileSkipLines, conf.inputFileHasHeader, conf.inputFileSkipDifferingLines, conf.inputFileNullString));

			ResultReceiver resultReceiver = new ResultCache("MetanomeMock", getAcceptedColumns(relationalInputGenerator));
			//ResultReceiver resultReceiver = new ResultCounter("MetanomeMock", getAcceptedColumns(relationalInputGenerator));
			
			hyFD.setRelationalInputConfigurationValue(HyFD.Identifier.INPUT_GENERATOR.name(), relationalInputGenerator);
			hyFD.setBooleanConfigurationValue(HyFD.Identifier.NULL_EQUALS_NULL.name(), Boolean.valueOf(conf.nullEqualsNull));
			hyFD.setBooleanConfigurationValue(HyFD.Identifier.VALIDATE_PARALLEL.name(), Boolean.valueOf(conf.validateParallel));
			hyFD.setBooleanConfigurationValue(HyFD.Identifier.ENABLE_MEMORY_GUARDIAN.name(), Boolean.valueOf(conf.enableMemoryGuardian));
			hyFD.setIntegerConfigurationValue(HyFD.Identifier.MAX_DETERMINANT_SIZE.name(), Integer.valueOf(conf.maxDeterminantSize));
			hyFD.setResultReceiver(resultReceiver);
			
			long time = System.currentTimeMillis();
			hyFD.execute();
			time = System.currentTimeMillis() - time;
			
			if (conf.writeResults) {
				String outputPath = conf.measurementsFolderPath + conf.inputDatasetName + File.separator;
				List<Result> results = null;
				if (resultReceiver instanceof ResultCache)
					results = ((ResultCache)resultReceiver).fetchNewResults();
				else
					results = new ArrayList<>();
				
				FileUtils.writeToFile(
						hyFD.toString() + "\r\n\r\n" + 
						"Runtime: " + time + "\r\n\r\n" + 
						"Results: " + results.size() + "\r\n\r\n" + 
						conf.toString(), outputPath + conf.statisticsFileName);
				FileUtils.writeToFile(format(results), outputPath + conf.resultFileName);
			}
		}
		catch (AlgorithmExecutionException e) {
			e.printStackTrace();
		}
		catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private static String format(List<Result> results) {
		HashMap<String, List<String>> lhs2rhs = new HashMap<String, List<String>>();

		for (Result result : results) {
			FunctionalDependency fd = (FunctionalDependency) result;
			
			StringBuilder lhsBuilder = new StringBuilder("[");
			Iterator<ColumnIdentifier> iterator = fd.getDeterminant().getColumnIdentifiers().iterator();
			while (iterator.hasNext()) {
				lhsBuilder.append(iterator.next().toString());
				if (iterator.hasNext())
					lhsBuilder.append(", ");
			}
			lhsBuilder.append("]");
			String lhs = lhsBuilder.toString();
			
			String rhs = fd.getDependant().toString();
			
			if (!lhs2rhs.containsKey(lhs))
				lhs2rhs.put(lhs, new ArrayList<String>());
			lhs2rhs.get(lhs).add(rhs);
		}
		
		StringBuilder builder = new StringBuilder();
		ArrayList<String> lhss = new ArrayList<String>(lhs2rhs.keySet());
		Collections.sort(lhss);
		for (String lhs : lhss) {
			List<String> rhss = lhs2rhs.get(lhs);
			Collections.sort(rhss);
			
			if (rhss.isEmpty())
				continue;
			
			builder.append(lhs + " --> ");
			builder.append(CollectionUtils.concat(rhss, ", "));
			builder.append("\r\n");
		}
		return builder.toString();
	}
}
