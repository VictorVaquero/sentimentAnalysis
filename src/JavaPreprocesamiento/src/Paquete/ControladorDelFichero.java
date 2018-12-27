package Paquete;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
public class ControladorDelFichero{
	private PrintWriter fichero;
	private FileWriter archivo;
	
	public ControladorDelFichero(){
		 try{
			archivo = new FileWriter(Prepro.DIR+"pruebaHilos.txt",true);
	        fichero = new PrintWriter(archivo);
		 } catch (Exception e) {
	        e.printStackTrace();
	    }
	}
	
	public synchronized void println(String a) {
		fichero.println(a);
		cerrar();
	}
	
	public void cerrar(){
		try {
			archivo.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
}