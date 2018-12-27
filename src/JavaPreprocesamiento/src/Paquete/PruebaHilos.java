package Paquete;

import java.util.List;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;

class PruebaHilos implements Runnable{
    
	
	private Thread hilo;
    private LexicalizedParser lp;
    private List<HasWord> sentence;
    
    

    public PruebaHilos(String nombre, LexicalizedParser lp, List<HasWord> sentence){
         hilo = new Thread(this,nombre);
         this.lp = lp;
         this.sentence = sentence;
    }
    
    public void run(){
    	ControladorDelFichero control = new ControladorDelFichero();
    	String texto = lp.apply(sentence).toString();
    	control.println(texto);
    }

	public Thread comenzar() {
    	hilo.start();
    	return hilo;
	}
     
}