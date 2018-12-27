package Paquete;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class Preprocesado {

	
	public static void main(String[] args) throws IOException {
		ArrayList<String> texto = new ArrayList<String>();
		try {
			BufferedReader b = new BufferedReader(new InputStreamReader(new FileInputStream(
					"/home/victor/workspace/python/sentimentAnalysis/JavaPreprocesamiento/ing.csv"), "utf-8"));
	        String cadena;
	        while((cadena = b.readLine())!=null) {
	        	texto.add(cadena.toLowerCase());
	        }
	        b.close();
		} catch (Exception e) {
	         
	    }
		
		ArrayList<String> reglas = new ArrayList<String>();
		reglas.add("ic$"); //regla 3
		reglas.add("[^aeiou0123456789]e$"); //regla 5
		reglas.add("ie$"); // regla 6
		reglas.add("l$"); //regla 4
		reglas.add("^([a-z]{0,2})?[aeiou]{1,2}[^aeiou0123456789y]$"); // regla 1
		reglas.add("[a-z][aeiou][^aeiou0123456789y]$"); //regla 2
		
		int control = 0;
		for(int i=0; i<texto.size();i++){
			control = 0;
			for(int j =0; j<reglas.size(); j++){
				if(control==0){
					Pattern regla = Pattern.compile(reglas.get(j));
					Matcher encaja = regla.matcher(texto.get(i));
					if(encaja.find()){
						if(j==0){
							String aux = texto.get(i) + "king";
							texto.set(i, aux);
							control++;
						}else if(j==1){
							if(!texto.get(i).equals("be")){
								int ultimaPosicion = texto.get(i).length();
								String aux = encaja.replaceAll(texto.get(i).charAt(ultimaPosicion-2) + "ing");
								texto.set(i, aux);
							}else{
								String aux = texto.get(i) + "ing";
								texto.set(i, aux);
							}
							control++;
						}else if(j==2){
							String aux = encaja.replaceAll("ying");
							texto.set(i, aux);
							control++;
						}else if(j==3 || j==4 || j==5){
							int ultimaPosicion = texto.get(i).length();
							String aux = texto.get(i) + texto.get(i).charAt(ultimaPosicion-1) + "ing";
							texto.set(i, aux);
							control++;
						}
					}
				}
			}
			if(control == 0){
				String aux = texto.get(i) + "ing";
				texto.set(i, aux);
			}
		}
		
		ArrayList<String> texto2 = new ArrayList<String>();
		try {
			BufferedReader b = new BufferedReader(new InputStreamReader(new FileInputStream(
					"/home/victor/workspace/python/sentimentAnalysis/JavaPreprocesamiento/ingRegu.csv"), "utf-8"));
	        String cadena;
	        while((cadena = b.readLine())!=null) {
	        	if(cadena.charAt(cadena.length()-1)=='e'){
	        		texto2.add(cadena.toLowerCase()+"d");
	        	}else{
	        		texto2.add(cadena.toLowerCase()+"ed");
	        	}
	        }
	        b.close();
		} catch (Exception e) {
	         
	    }
		
		
		for(int i=0; i<texto.size();i++){
			System.out.println(texto2.get(i));
		}
	}
	
}