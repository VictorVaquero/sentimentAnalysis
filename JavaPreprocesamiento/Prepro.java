package Paquete;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;

public class Prepro {

	
	public static void main(String[] args) throws IOException {
		String texto = "";
		try {
			BufferedReader b = new BufferedReader(new InputStreamReader(new FileInputStream("D:/Domadoro/Documents/General/Universidad/Tercero/ALGC/Proyecto/market.csv"), "utf-8"));
	        String cadena;
	        int contador = 0;
	        while((cadena = b.readLine())!=null) {
	        	if(!(contador==0)){
	        		texto = texto + cadena.toLowerCase();
	        	}
	        	contador++;
	        }
	        b.close();
		} catch (Exception e) {
	         
	    }

		String[] signos = {" ½ "," £ ","\""," ‘ "," ’ ", " of ", " the "," an "," a ",",",".",";",":","...","?","!","¡","(",")","[","]","-","_","/","'\'","*","$","@","0","1","2","3","4","5","6","7","8","9","·","#","|","º","ª","~","%","€","&","¬","=","^","+","¨","ç","<",">","•","”","“","—"};
		for(int i=0; i<signos.length;i++){
			texto = texto.replace(signos[i]," ");
		}
		
		texto = texto.replace("n't "," not ");
		texto = texto.replace("n´t "," not ");
		texto = texto.replace("n`t "," not ");
		
		// de plural a singular
		texto = texto.replace("ies ","y ");
		texto = texto.replace("s "," ");
		texto = texto.replace("es "," "); // no siempre ejemplo Species
		texto = texto.replace("zzes ","z ");
		texto = texto.replace("ves ","f "); //no siempre lives --> life

		
		//cargamos los verbos regulares
		try {
			BufferedReader b = new BufferedReader(new InputStreamReader(new FileInputStream("D:/Domadoro/Documents/General/Universidad/Tercero/ALGC/Proyecto/RegularVerbs.csv"), "utf-8"));
	        String cadena;
	        String[] separada;
	        while((cadena = b.readLine())!=null) {
	        	separada = cadena.split(",");
	        	texto = texto.replaceAll(" " + separada[1] + " "," " + separada[0] + " ");
	        }
	        b.close();
		} catch (Exception e) {
	         
	    }
		
		//cargamos los verbos irregulares
		try {
			BufferedReader b = new BufferedReader(new InputStreamReader(new FileInputStream("D:/Domadoro/Documents/General/Universidad/Tercero/ALGC/Proyecto/IrregularVerbs.csv"), "utf-8"));
	        String cadena;
	        String[] separada;
	        while((cadena = b.readLine())!=null) {
	        	separada = cadena.split(",");
	        	for(int i = 1; i<separada.length;i++){	
	        		texto = texto.replaceAll(" " + separada[i] + " "," " +separada[0] + " ");
	        	}
	        }
	        b.close();
		} catch (Exception e) {
	         
	    }
		
		
		FileWriter fichero = null;
        PrintWriter pw = null;
        try{
            fichero = new FileWriter("D:/Domadoro/Documents/General/Universidad/Tercero/ALGC/Proyecto/market.txt");
            pw = new PrintWriter(fichero);
            pw.println(texto);
            fichero.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
	}
}
