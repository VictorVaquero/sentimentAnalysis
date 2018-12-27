package Paquete;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;


public class Prepro {
	
	/*
	static String DIR = "/home/victor/workspace/python/sentimentAnalysis/JavaPreprocesamiento/";
	static String FILE = "market.csv";
	static String FILE2 = "market2.txt";
	static String FINAL = "listaPalabrasClave2.txt";
	*/
	static boolean SENT = false;  // Preprocesamos la base de datos con etiquetas o tweets nuestros

	public static String DIR =
			SENT ? "/home/victor/workspace/python/sentimentAnalysis/database/" : "/home/victor/workspace/python/sentimentAnalysis/database/google/";
	
	static String FILE;
	static String SUFIX_HALF = "_medio_limpios.txt";
	static String SUFIX_FINAL = "_limpios.txt";
	static String END = ".csv";
	static int SIZE_TEXT = 5; // lee x tweets por vez
	static int TWEET_COLUM = 5;
	static int TWEET_DATE_COLUM = 1;
	
	
		
	static String SENT_FILE = "training.1600000.processed.noemoticon_random.csv"; 
	static int SENT_TWEET_COLUM = 5;
	static int SENT_LABEL_COLUM = 0;
	static int SENT_DATE_COLUM = 2;
	
	
	public static void main(String[] args) throws IOException {
		long startTime = System.currentTimeMillis();
		//int n = 0;
		File[] listOfFiles;
		
		System.out.println("Procesing SENT? "+ SENT);
		if(SENT)
			listOfFiles = new File[] {new File(DIR+SENT_FILE)}; 
		else
			listOfFiles = getListaArchivos();
		
	
		
		for (File file : listOfFiles) {
			System.out.println(file);
			
		    if (file != null && file.isFile()) {
		    	System.out.println(file.isFile());
		        FILE = file.getName();
		        System.out.println(file);
		        
		        BufferedReader b = new BufferedReader(new InputStreamReader(new FileInputStream(
						DIR+FILE), "utf-8"));
				String[] texto = new String[SIZE_TEXT];
				String[] label = new String[SIZE_TEXT];
				String[] date = new String[SIZE_TEXT];
				
				while(true) {
					int contador = 0;
					try {
						 String cadena;
					     while(contador < SIZE_TEXT) {
					    	cadena = b.readLine(); // Si no hay mas lineas, revienta y sales del while
					      	texto[contador] = cadena.toLowerCase().split(",")[SENT ? SENT_TWEET_COLUM : TWEET_COLUM];
					      	date[contador] = cadena.toLowerCase().split(",")[SENT ? SENT_DATE_COLUM : TWEET_DATE_COLUM];
					      	if(SENT) {
					      		label[contador] = cadena.toLowerCase().split(",")[SENT_LABEL_COLUM];
					      		
					      	}
					       	contador++;
					    }
					    //n = contador;
					    
					} catch (Exception e) {
						System.out.println("----- Fin archivo ----- " + e);
					    break;
					}
						
					System.out.println("\n\nLeer "+ FILE + " hecho --> tamaño: "+ contador);
					System.out.println("t[0]: "+texto[0]);
					
					reglasBasicas(texto);
				
			        System.out.println("Reglas basicas hecho");
				
			        
			        // ----------------------------------------- Primer fichero medio limpio --------------
				    String nombre = DIR+FILE.substring(0, FILE.length()-4)+SUFIX_HALF;
			        FileWriter fichero = null;
				    PrintWriter pw = null;     	
		    	    try{
		    	    	System.out.println("Escribe en "+nombre);
		                fichero = new FileWriter(nombre);
		                pw = new PrintWriter(fichero);
		                for(String t:texto)  pw.println(t);
		                fichero.close();
		            } catch (Exception e) {
		                e.printStackTrace();
		            }		 
		    	    System.out.println("t[0]: "+texto[0]);
		    	    // ---------------------------------------- Normalizado del texto -------------------
		    	    
		    		texto = ParserDemo.demoDP(nombre, texto,startTime);
		    	
		    		// ----------------------------------------- Escritura del fichero ya limpio --------------
		    		fichero = null;
			        pw = null;
			        try{
			        	System.out.println("Fin archivo append +tam : " + texto.length);
			        	String FILE_LAST = DIR+FILE.substring(0, FILE.length()-4)+SUFIX_FINAL;
				        File f = new File(FILE_LAST);
				        f.createNewFile();
			            fichero = new FileWriter(FILE_LAST,true);
			            pw = new PrintWriter(fichero);
			            //pw.println(palabrasClave);
			            for(int ti=0;ti<texto.length;ti++) {
			            	String t = texto[ti];
					        //System.out.println(texto);
					        t = t.replace(". "," ");
					        t = t.replace(".","");
					        while(t.contains("  ")){
					        	t= t.replace("  "," ");
					        }
					        pw.println(SENT ? date[ti] +","+ label[ti] + "," + t  : date[ti] + "," + t);
			            }
			            fichero.close();
			        } catch (Exception e) {
			            e.printStackTrace();
			        }		   
		        
		   
		        
				}
				b.close();
		    }
		}
	}
	
	static private File[] getListaArchivos() {
		File folder = new File(DIR);
		File[] listOfFiles = folder.listFiles();
		for(int x=0;x<listOfFiles.length;x++) {
			String name = listOfFiles[x].getName();
			if(!name.substring(name.length()-END.length(), name.length()).equals(END))
				listOfFiles[x] = null;
		}
		
		return listOfFiles;
	}

	static private void reglasBasicas(String[] texto) {

		for(int t=0;t<texto.length;t++) {
			texto[t] = texto[t].replace("n‘t "," not ");
			texto[t] = texto[t].replace("n’t "," not ");
			texto[t] = texto[t].replace("n't "," not ");
			texto[t]= texto[t].replace("n´t "," not ");
			texto[t] = texto[t].replace("n`t "," not ");
			texto[t] = texto[t].replace(" ca "," can ");
			
			texto[t] = texto[t].replace(".com "," ");
			texto[t] = texto[t].replace(" www."," ");
			
			texto[t] = texto[t].replace("‘s "," ");
			texto[t] = texto[t].replace("’s "," ");
			texto[t] = texto[t].replace("'s "," ");
			texto[t] = texto[t].replace("´s "," ");
			texto[t] = texto[t].replace("`s "," ");
			
			String[] toBe = {"’m ","`m ","´m ","'m ","‘m ","’re ","`re ","´re ","'re ","‘re "," are "," is "};
			for(int i=0; i<toBe.length;i++){
				texto[t] = texto[t].replace(toBe[i]," be ");
			}
			
			String[] toHave = {"’ve ","`ve ","´ve ","'ve ","‘ve ","’d ","`d ","´d ","'d ","‘d "};
			for(int i=0; i<toHave.length;i++){
				texto[t] = texto[t].replace(toHave[i]," have ");
			}
			
			String[] will = {"’ll ","`ll ","´ll ","'ll ","‘ll "};
			for(int i=0; i<will.length;i++){
				texto[t] = texto[t].replace(will[i]," will ");
			}
			
			
			
			
			texto[t]= texto[t].replace(". "," . ");
			
			/*
			String[] signos = {"©","’ "," ½ "," £ "," £ ","\""," ‘"," ‘ "," ’ ",",",";",":",
					"...","…","?","!","¡","(",")","[","]","-","_","/","'\'","*","$","@","0",
					"1","2","3","4","5","6","7","8","9","·","#","|","º","ª","~","%","€","&",
					"¬","=","^","+","¨","ç","<",">","•","”","“","—","う","", "'","×","*",",","≠","卍","×"};
			for(int i=0; i<signos.length;i++){
				texto[t] = texto[t].replace(signos[i]," ");
			}
			*/
			texto[t] = texto[t].replaceAll("[^a-z ]", "");
	        
	        while(texto[t].contains("  ")){
	        	texto[t]= texto[t].replace("  "," ");
	        }
		}
	}
		
}
