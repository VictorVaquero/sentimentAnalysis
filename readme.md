# Natural Language Processing for Finance

Investment prototype for an Stock Exchange through public opinion analysis. We try to detect underlying patterns in market movements correlated with positive and negative sentiment about a target business.

## Getting Started

Use git clone to get the source. Everything but the preprocessing step and the final prediction, which have been coded in Java, is in Python.

### Prerequisites

The data recollection uses multiple third-party library's: GetOldTweets-python, TensorFlow and Stanford NLP. You may need to install them.

## Structure
* src - Actual code
* database - Main project database
* store - Saved embedding and prediction models
* eventos/eventos_sent - TensorBoard files

## Src
*  javaPreprocesamiento - Source for processing data
*  embeddings.py - Code for training the first neural network
* embeddingsMatrix.py - Simple neural network to obtain embeddings from a model
* sentPrediction.py - Code for training the second neural network
* randomize.py - Utility to randomize a file
* MostrarGrafica.py - Utility to create a final prediction's graphic



## Built With

* [GetOldTweets](https://github.com/Jefferson-Henrique/GetOldTweets-python) - Twitter Parser
* [Stanford NLP](https://nlp.stanford.edu/) - Syntax Parser 
* [TensorFlow](https://www.tensorflow.org/) - Neural Networks


## Authors

* **Victor Vaquero Martinez** - *Main work* - [Victor](https://github.com/VictorVaquero/)
* **Juan Rodríguez** - *Main work* - [Juan]()

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details



