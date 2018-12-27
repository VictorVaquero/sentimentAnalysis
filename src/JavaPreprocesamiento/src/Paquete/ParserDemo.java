package Paquete;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;

class ParserDemo {
	
	private static int MAX_THREAD = 1;
	
	
	public static String[] demoDP(String filename, String[] contenido, long startTime) {
		
		ArrayList<String> repetidas = new ArrayList<String>();
		String parserModel = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";
		LexicalizedParser lp = LexicalizedParser.loadModel(parserModel);

	    TreebankLanguagePack tlp = lp.treebankLanguagePack();
	    GrammaticalStructureFactory gsf = null;
	    if (tlp.supportsGrammaticalStructures()) {
	    	gsf = tlp.grammaticalStructureFactory();
	    }
	    
	    File x = new File(Prepro.DIR + "pruebaHilos.txt");
		x.delete();
		//long endTime = System.currentTimeMillis() - startTime;
	    //System.out.println(endTime);
	    //startTime = System.currentTimeMillis();
		int contador = 0;
		Thread[] hilos = new Thread[MAX_THREAD];
	    for (List<HasWord> sentence : new DocumentPreprocessor(filename)) {
	    	System.out.println("Lanza thread #"+contador);
	    	PruebaHilos nombre =new PruebaHilos("#"+contador,lp,sentence);
	    	hilos[contador] = nombre.comenzar();
	    	contador++;
	    	if(contador == MAX_THREAD) {
	    		for(Thread t:hilos) {
					try {
						t.join();
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
	    		System.out.println("--");
	    		contador = 0;
	    	}   	
	    }
	    
	    for(Thread t:hilos) {
			try {
				t.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			
	    }
	    System.out.println("Fuera threads");
	    //endTime = System.currentTimeMillis() - startTime;
	    //System.out.println(endTime);
	    //startTime = System.currentTimeMillis();
		try {
			BufferedReader b = new BufferedReader(new InputStreamReader(new FileInputStream(Prepro.DIR+"pruebaHilos.txt"), "utf-8"));
	        String texto;
	        while((texto = b.readLine())!=null) {
	        	
		    	//Eliminar determinantes
		    	Pattern regla = Pattern.compile("(DT [a-z]*)");
				Matcher encaja = regla.matcher(texto);
				while(encaja.find()){
					String[] palabra = encaja.group(1).split(" ");
					for(int i=0;i<contenido.length;i++)
						contenido[i] = contenido[i].replace(" " + palabra[1] + " ", " ");
				}
				
				//Eliminar CC
		    	regla = Pattern.compile("(CC [a-z]*)");
				encaja = regla.matcher(texto);
				while(encaja.find()){
					String[] palabra = encaja.group(1).split(" ");
					for(int i=0;i<contenido.length;i++)
						contenido[i] = contenido[i].replace(" " + palabra[1] + " ", " ");
				}
				
				//Eliminar TO
		    	regla = Pattern.compile("(TO [a-z]*)");
				encaja = regla.matcher(texto);
				while(encaja.find()){
					String[] palabra = encaja.group(1).split(" ");
					for(int i=0;i<contenido.length;i++)
						contenido[i] = contenido[i].replace(" " + palabra[1] + " ", " ");
				}
				
				//Eliminar IN
				regla = Pattern.compile("(IN [a-z]*)");
				encaja = regla.matcher(texto);
				while(encaja.find()){
					String[] palabra = encaja.group(1).split(" ");
					for(int i=0;i<contenido.length;i++)
						contenido[i] = contenido[i].replace(" " + palabra[1] + " ", " ");
				}
				
		    	//Buscar verbos
		    	regla = Pattern.compile("(VB[A-MOQ-Z] [a-z]*)");
				encaja = regla.matcher(texto);
				while(encaja.find()){
					String[] palabra = encaja.group(1).split(" ");
					if(!repetidas.contains(palabra[1]) && !palabra[1].equals("i") && !palabra[1].equals("he") && !palabra[1].equals("she") && !palabra[1].equals("it")){
						repetidas.add(palabra[1]);
					}
				}
				
				//Buscar plurales
				regla = Pattern.compile("(NNS [a-z]*)");
				encaja = regla.matcher(texto);
				while(encaja.find()){
					String[] palabra = encaja.group(1).split(" ");
					if(!repetidas.contains(palabra[1]) && !palabra[1].equals("i") && !palabra[1].equals("he") && !palabra[1].equals("she") && !palabra[1].equals("it")){
						repetidas.add(palabra[1]);
					}
				}
	        }
	        b.close();
		} catch (Exception e) {
	         
	    }
		
	    //endTime = System.currentTimeMillis() - startTime;
	    //System.out.println(endTime);
	    //startTime = System.currentTimeMillis();
	    //System.out.println(repetidas.size());
	    for(int i=0; i<repetidas.size();i++){
		    String url = "https://en.wiktionary.org/wiki/" + repetidas.get(i);
			//System.out.println(url);
		    try{
		    	Document document = getHtmlDocument(url);
		    	Elements entradas = document.select("div.mw-parser-output");
		    	for (Element elem : entradas) {
		    		String titulo = elem.getElementsByClass("form-of-definition-link").text();
		    		for(int u=0;u<contenido.length;u++)
		    			contenido[u] = contenido[u].replace(repetidas.get(i), titulo.split(" ")[0]);
		    	}
		    }catch(RuntimeException e){
		    	//System.out.println("Palabra perdida: " + repetidas.get(i));
		    }
	    }
	    for(int u=0;u<contenido.length;u++)
	    	contenido[u] = contenido[u].replace(".","");
	    long endTime = System.currentTimeMillis() - startTime;
	    System.out.println(endTime);
	    return contenido;
	}
	
	/**
	 * Con este método devuelvo un objeto de la clase Document con el contenido del
	 * HTML de la web que me permitirá parsearlo con los métodos de la librelia JSoup
	 * @param url
	 * @return Documento con el HTML
	 */
	public static Document getHtmlDocument(String url) {

	    Document doc = null;
		try {
	    	doc = Jsoup.connect(url).userAgent("Mozilla/5.0").get();
	    } catch (IOException ex) {
	    }
	    return doc;
	}

}