from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))

#Definition of streaming class

from threading import Thread

class StreamingThread(Thread):
    def __init__(self, ssc):
        Thread.__init__(self)
        self.ssc = ssc
    def run(self):
        ssc.start()
        ssc.awaitTermination()
    def stop(self):
        print('----- Stopping... this may take a few seconds -----')
        self.ssc.stop(stopSparkContext=False, stopGraceFully=True)
        
sc
spark

# load data

df = spark.read.format("csv").options(header='true', delimiter=";", inferSchema='true').load("merged6.csv")

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression

regexTokenizer = RegexTokenizer(inputCol="review_text", outputCol="words", pattern="\\W")

add_stopwords = ["the"] 
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)

countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=1)

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

label_stringIdx = StringIndexer(inputCol = "review_score", outputCol = "label")

locale = sc._jvm.java.util.Locale
locale.setDefault(locale.forLanguageTag("en-US"))

pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx])

pipelineFit = pipeline.fit(df)

dataset = pipelineFit.transform(df)

def myCleaner(df):
    
    clean_data = pipelineFit.transform(df)
    return clean_data

(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))

lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainingData)

from pyspark.streaming import StreamingContext
from pyspark.sql import Row
from pyspark.sql.functions import udf, struct
from pyspark.sql.types import IntegerType

globals()['models_loaded'] = False

def process(time, rdd):
    if rdd.isEmpty():
        return
    print("========= %s =========" % str(time))
    
    # Convert to data frame
    df = spark.read.json(rdd)
    df.show()
    
    clean_data = myCleaner(df)
    
    # Load in the model if not yet loaded:
    if not globals()['models_loaded']:
        # load in your models here
        globals()['my_model'] = 'lrModel' # 'my_model' # Replace this with:    [...].load('my_logistic_regression')
        globals()['models_loaded'] = True
        
    # Predict using the model:
    df_result =  lrModel.transform(clean_data)
    df_result.select("book_title","review_score","probability","label","prediction").show()
    
    
ssc = StreamingContext(sc, 10)

lines = ssc.socketTextStream("seppe.net", 7778)
lines.foreachRDD(process)

ssc_t = StreamingThread(ssc)
ssc_t.start()

# Streaming begins, only run ssc_t.stop() to stop

ssc_t.stop()