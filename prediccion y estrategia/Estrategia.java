package Paquete;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.sql.Date;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Collections;
import java.util.Hashtable;

public class Estrategia {

	public static void main(String[] args) {
		String url = "ASensi.csv"; // El fichero con el resultado del analisis de los tweets
		String urlHistorico = "GOOGL"; // El fichero con el historico de precios de esa empresa en formato csv pero la url sin formato porque se usara luego para dar nombre al nuevo fichero
		String[] texto = new String[4];
		float[] numero = new float[2]; // posicion 0 valores negativos de los tweets, 1 valores positivos
		try {
			BufferedReader b = new BufferedReader(new InputStreamReader(new FileInputStream("D:/Domadoro/Documents/General/Universidad/Tercero/ALGC/Proyecto/" + url), "utf-8"));
	        String cadena;
	        int contador = 0;
	        while((cadena = b.readLine())!=null) {
	        	if(!(contador==0)){
        			texto = cadena.toLowerCase().split(",");
    	        	try{
    	        		for(int i = 0; i<numero.length; i++){
    	        			numero[i] = numero[i] + Float.parseFloat(texto[i]);
    	        		}
    	        	}catch(Exception e){
    	        		System.err.println("Problema a la hora de parsear el los numeros");
    	        	}
	        	}
	        	contador++;
	        }
	        b.close();
		} catch (Exception e) {
	         System.out.println("Problema al leer el fichero");
	    }
		
		if(numero[0]>numero[1]){
			// usar numero[0];
		}else{
			// usar numero[1];
		}
		
		// Suavizado
		ArrayList<String[]> precios = new ArrayList<String[]>();
		try {
			BufferedReader b = new BufferedReader(new InputStreamReader(new FileInputStream("D:/Domadoro/Documents/General/Universidad/Tercero/ALGC/Proyecto/" + urlHistorico + ".csv"), "utf-8"));
	        String cadena;
	        int contador = 0;
	        while((cadena = b.readLine())!=null) {
	        	if(!(contador==0)){
        			texto = cadena.toLowerCase().split(",");
        			precios.add(texto);
	        	}
	        	contador++;
	        }
	        b.close();
		} catch (Exception e) {
	         System.out.println("Problema al leer el fichero");
	    }
		
		// Prediccion
		
		String[] ultimaFecha = texto[0].split("-"); // ultima fecha
		Calendar fecha = Calendar.getInstance();
		fecha.set(Integer.parseInt(ultimaFecha[0]), Integer.parseInt(ultimaFecha[1])-1, Integer.parseInt(ultimaFecha[2]));
		
		
		double[] prediccion = prediccion(precios); // guardar las prediccion para close
		
		PrintWriter fichero;
		FileWriter archivo;
		try{
			archivo = new FileWriter("D:/Domadoro/Documents/General/Universidad/Tercero/ALGC/Proyecto/" + urlHistorico + "Prediccion.csv",false);
	        fichero = new PrintWriter(archivo);
	        fichero.println("date,high,low,close");
	        for(int i = 0; i<precios.size();i++){
	        	fichero.println(precios.get(i)[0] + "," + precios.get(i)[2] + "," + precios.get(i)[3] + "," + precios.get(i)[4]);
	        }
	        System.out.println("Prediccion:");
	        for(int i = 0; i<prediccion.length; i++){
	        	fecha.add(Calendar.DAY_OF_YEAR, 1);
	        	fichero.println(fecha.get(Calendar.YEAR) + "-" + Integer.sum(fecha.get(Calendar.MONTH),1) + "-" + fecha.get(Calendar.DAY_OF_MONTH) + "," + texto[2] + "," + texto[3] + "," + prediccion[i]);
	        	if(Double.parseDouble(texto[2]) <= prediccion[i]){
	        		System.out.print(fecha.get(Calendar.YEAR) + "-" + Integer.sum(fecha.get(Calendar.MONTH),1) + "-" + fecha.get(Calendar.DAY_OF_MONTH));
	        		System.out.println(" Subira por encima del high: Invierte");
	        	}else if(Double.parseDouble(texto[3]) >= prediccion[i]){
	        		System.out.print(fecha.get(Calendar.YEAR) + "-" + Integer.sum(fecha.get(Calendar.MONTH),1) + "-" + fecha.get(Calendar.DAY_OF_MONTH));
	        		System.out.println(" Bajara por debajo del low: No inviertas");
	        	}
	        }
	        archivo.close();
		} catch (Exception e) {
	        e.printStackTrace();
	    }

		
	}

	public static double[] prediccion(ArrayList<String[]> precios) {
		int k = 1; // se puede quitar
		int periodosFuturos = 50; // Numero de dias que tarda en repetirse el ciclo de cada periodo, tambien es numero de dias que predice en el futuro
		double[] prediccion = new double[periodosFuturos]; // Donde se guarda la prediccion
		
		// nuestra estimacion
		double alpha1 = 0.1;
		double alpha2 = 0.2;
		double alpha3 = 0.7;
		
		// Posicion 0 Periodo actual // Posicion 1 Periodo anterior
		double[] nivel = new double[2];
		double[] tendencia = new double[2];
		ArrayList<Double> St = new ArrayList<Double>(); // Estimado de estacionalidad 
		double[] hisPre = new double[2]; // Historico de precios
		double[] proFut = new double[2]; // Pronostico de P periodos futuros
		
		for(int i = 0; i<precios.size(); i++){
			if(i == 0){ // primer periodo
				hisPre[0] = Double.parseDouble(precios.get(i)[4]);
				nivel[0] = hisPre[0];
				tendencia[0] = 0;
				for(int j = 0; j<periodosFuturos+1;j++){
					St.add(1.0);
				}
			}else{
				hisPre[1] = hisPre[0];
				hisPre[0] = Double.parseDouble(precios.get(i)[4]);
				nivel[1] = nivel[0];
				tendencia[1] = tendencia[0];
				nivel[0] = alpha1*(hisPre[0]/St.get(0)) + (1 - alpha1)*(nivel[1] + tendencia[1]);
				tendencia[0] = alpha2*(nivel[0] - nivel[1]) + (1 - alpha2)*tendencia[1];
				proFut[0] =(nivel[1] + k*tendencia[1])*St.get(i-1);
				St.add(alpha3*(hisPre[0]/nivel[0]) + (1-alpha3)*St.get(i-1));
			}
		}
		
		for(int j = 0; j<periodosFuturos;j++){
			prediccion[j] = (nivel[1] + tendencia[1])*St.get(precios.size()+j);
		}
		return prediccion;
	}
}
