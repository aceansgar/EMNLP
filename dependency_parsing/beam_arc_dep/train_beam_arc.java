package hanlp_dep;

import java.io.IOException;

import com.hankcs.hanlp.*;
import com.hankcs.hanlp.corpus.dependency.CoNll.*;
import com.hankcs.hanlp.dependency.*;
import com.hankcs.hanlp.dependency.nnparser.*;
import com.hankcs.hanlp.dependency.perceptron.parser.KBeamArcEagerDependencyParser;
import com.hankcs.hanlp.utility.*;
import com.hankcs.hanlp.dependency.perceptron.parser.Main;
import com.hankcs.hanlp.dependency.perceptron.transition.parser.KBeamArcEagerParser;
import com.hankcs.hanlp.dependency.perceptron.accessories.Evaluator;
import com.hankcs.hanlp.dependency.perceptron.accessories.Options;

public class train_beam_arc 
{
	static void train_conllu(String[] args)
	{
		try
		{
			Options options=new Options();
			options.train=true;
			options.inputFile="data/train.conllu";
			options.trainingIter=2;
			// path of test file, name of result file is based on it
			// if we do not set test path, it will only train the model
			options.devPath="data/test.conllu";
			options.evaluate=true;
			options.scorePath="data/res_score.txt";
			// store the trained model to this file
			options.modelFile="data/my_beam_arc_model.bin";
			options.inputFile=args[1];
			options.devPath=args[2];
			options.trainingIter=Integer.valueOf(args[3]);
			Main.train(options);
			return;
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}
	
	static void test_conllu(String[] args)
	{
		try
		 {
			 Options options=new Options();
			 // path of test file
			 options.devPath="data/test.conllu";
			 String output_path="data/test_output.conllu";
			 options.modelFile="data/my_beam_arc_model.bin";
			 options.scorePath="data/my_score.bin";
			 
			 options.devPath=args[1];
			 
			 KBeamArcEagerParser parser=new KBeamArcEagerParser(options.modelFile);
			 parser.parseConllFile(options.devPath, output_path, options.rootFirst, 
					 options.beamWidth, true, true, options.numOfThreads, false, options.scorePath);
			 double[] score = Evaluator.evaluate(options.devPath, output_path, options.punctuations);
			 System.out.printf("UAS=%.2f LAS=%.2f", score[0], score[1]);
            parser.shutDownLiveThreads();

		 }
		 catch(Exception e)
		 {
			 e.printStackTrace();
		 }
	}
	
	public static void main(String[] args) throws Exception
	{
		
		
//		String[] strings=new String[1];
//		strings[0]="--help";
//		Options.showHelp();
//		options=Options.processArgs(strings);
//		PerceptronParserModelPath = "data/model/dependency/perceptron.bin";
		
		try
		{
			String first_arg=args[0];
			// "train" means train+test
			if(first_arg.equals("train"))
				train_conllu(args);
			else if(first_arg.equals("test"))
				test_conllu(args);
				
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}

		
//		int arg_cnt=args.length;
//		if(arg_cnt!=3)
//		{
//			System.err.println("error: need 3 arguments:\n train_path test_path iter_num");
//		}
//		System.out.print("args: ");
//		for(int i=0;i<arg_cnt;++i)
//		{
//			System.out.printf("%d ", args[i]);
//		}
		

	}
}
