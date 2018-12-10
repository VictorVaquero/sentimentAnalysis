package Paquete;

import java.util.List;

import javax.imageio.IIOException;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;

public class MiHilo implements Runnable {
	Thread hilo;
	private LexicalizedParser lp;
	private List<HasWord> sentence;
	private String texto = "";
	private boolean continuar = true;

	MiHilo(String nombre){
		hilo = new Thread(this,nombre);
	}
	
	public MiHilo(LexicalizedParser lp) {
		this.lp=lp;
	}

	public void run(){
		while(continuar){
			try{
				texto = lp.apply(sentence).toString();
			}catch(Exception e){
				
			}
		}
    }
	
	public void detenElHilo(){
		continuar = false;
	}

	public void setSentencia(List<HasWord> sentence) {
		this.sentence = sentence;
	}

	public String getTexto() {
		return texto;
	}
	
	public static MiHilo crearYComenzar (String nombre){
		MiHilo miHilo = new MiHilo(nombre);
		miHilo.hilo.start();
        return miHilo;
    }
	
}
	

