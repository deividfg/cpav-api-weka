package br.com.softplan.ungp.cpav.weka;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Random;

public class TreeJ48 {
	public void testWekaJ48() throws Exception {

		//Dados
		Instances data = getData("/tramitacoes.arff", 1);
		System.out.println(data.numInstances() + " Registros");

		// Algoritimo J48
		J48 j48 = new J48();

		// Opcoes do J48
		String[] options = new String[4];
		options[0] = "-C"; // Definição do limite de confianca para a remocao.
		options[1] = "0.25"; // Valor do limite de confianca.
		options[2] = "-M"; // Definir o numero minimo de instancias por folha.
		options[3] = "2"; // Valor do numero minimo de instancias.
		j48.setOptions(options);

		//Deixando somente o cdSetoranterior = 10, para testes
		RemoveWithValues filter = new RemoveWithValues();

		String[] optionsFilter = new String[5];
		optionsFilter[0] = "-C"; // Escolha do atributo a ser usado na selecao
		optionsFilter[1] = "9"; // Indice do atributo usado na selecao
		optionsFilter[2] = "-L"; // Escolha do valor do atributo a ser usado na selecao
		optionsFilter[3] = "8"; // Indice do valor usado na selecao
		optionsFilter[4] = "-V"; // Inversão da seleção, ou seja, remove todos os outros valores.
		filter.setOptions(optionsFilter);

		filter.setInputFormat(data);
		Instances newData = Filter.useFilter(data, filter);

		// Construindo o classificador
		j48.buildClassifier(newData);

		Integer numIterations = 10; // Numero de iteracoes do crossValidator
		Random randData = new Random(1); // indice do gerador de numeros aleatorios
		Evaluation evalTree = evalModel(j48, newData, numIterations, randData);
		System.out.println("Resultado: \n" + evalTree.toSummaryString());

	}

	/** Cria um objeto de avaliacao que aplica o classificador aos dados fornecidos.
	 *
	 * @param classifier classificador
	 * @param data dados
	 * @param numberIterations Numero de iteracoes do crossValidator
	 * @param randData indice do gerador de numeros aleatorios
	 * @return objeto de avaliacao
	 */
	private Evaluation evalModel(
			Classifier classifier, Instances data, Integer numberIterations, Random randData ) throws Exception {
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(classifier, data, numberIterations, randData);
		return eval;
	}

	/** Leitura dos dados
	 *
	 * @param filename caminho e nome do arquivo
	 * @param posClass indice baseado na definicao de classe vista no final da lista de atributos.
	 * @return objeto de instancia
	 */
	private Instances getData( String filename, Integer posClass ) throws IOException, URISyntaxException {
		File file = new File(TreeJ48.class.getResource(filename).toURI());
		BufferedReader inputReader = new BufferedReader(new FileReader(file));
		Instances data = new Instances(inputReader);
		data.setClassIndex(data.numAttributes() - posClass);

		return data;
	}
}
