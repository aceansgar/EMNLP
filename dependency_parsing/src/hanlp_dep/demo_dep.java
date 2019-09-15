package hanlp_dep;
import java.io.IOException;

import com.hankcs.hanlp.*;
import com.hankcs.hanlp.dependency.*;
import com.hankcs.hanlp.dependency.nnparser.*;
import com.hankcs.hanlp.dependency.perceptron.accessories.Options;
import com.hankcs.hanlp.dependency.perceptron.accessories.Evaluator;
import com.hankcs.hanlp.dependency.perceptron.parser.KBeamArcEagerDependencyParser;
import com.hankcs.hanlp.utility.*;
import com.hankcs.hanlp.dependency.perceptron.transition.parser.KBeamArcEagerParser;
import com.hankcs.hanlp.corpus.io.IOUtil;

public class demo_dep 
{
	 public static void main(String[] args) 
	    {
		 	 
//		 	try
//		 	{
//		 		IDependencyParser parser = new KBeamArcEagerDependencyParser().enableDeprelTranslator(false);
//
//		 		CoNLLSentence sentence=parser.parse("�����������������ȷ���˰ѻ���ӥ���������ȸ��Ϊ����Ŀ�ꡣ");
//		 		System.out.println(sentence);
//			 	
//	
//		        // ���Է���ر�����
//		        for (CoNLLWord word : sentence)
//		        {
//		            System.out.printf("%s --(%s)--> %s\n", word.LEMMA, word.DEPREL, word.HEAD.LEMMA);
//		        }
//		        // Ҳ����ֱ���õ����飬����˳����������
//		        CoNLLWord[] wordArray = sentence.getWordArray();
//		        for (int i = wordArray.length - 1; i >= 0; i--)
//		        {
//		            CoNLLWord word = wordArray[i];
//		            System.out.printf("%s --(%s)--> %s\n", word.LEMMA, word.DEPREL, word.HEAD.LEMMA);
//		        }
//		        // ������ֱ�ӱ�����������ĳ��������ĳ���ڵ�һ·���������
//		        CoNLLWord head = wordArray[12];
//		        while ((head = head.HEAD) != null)
//		        {
//		            if (head == CoNLLWord.ROOT) System.out.println(head.LEMMA);
//		            else System.out.printf("%s --(%s)--> ", head.LEMMA, head.DEPREL);
//		        }
//		 	}
//		 	catch(Exception e)
//		 	{
//		 		e.printStackTrace();
//		 	}
		 try
		 {
			 Options options=new Options();
			 // path of test file
			 options.devPath="data/test.conllu";
			 String output_path="data/test_output.conllu";
			 options.modelFile="data/my_beam_arc_model.bin";
			 options.scorePath="data/my_score.bin";
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
}
