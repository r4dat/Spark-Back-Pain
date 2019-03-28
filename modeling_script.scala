import org.apache.spark.sql.types._
import org.apache.spark.ml.{Pipeline,PipelineModel,PipelineStage}
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer,VectorAssembler,CountVectorizer,CountVectorizerModel}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.classification.{LogisticRegression,RandomForestClassifier,GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
spark.conf.set("spark.sql.broadcastTimeout",(1*60)) // so if we hit etgDx bug it won't take forever to timeout.

var mbr = spark.read.format("csv").option("header", "true").load("/user/mschlomka/backpain/FINAL_mbr.csv")
var diag = spark.read.format("csv").option("header", "true").load("/user/mschlomka/backpain/FINAL_diag.csv")
var medical = spark.read.format("csv").option("header", "true").load("/user/mschlomka/backpain/FINAL_medical.csv")
var rx = spark.read.format("csv").option("header", "true").load("/user/mschlomka/backpain/FINAL_rx.csv")
var scoringDat = spark.read.format("csv").option("header", "true").load("/user/mschlomka/backpain/scoringData.csv")


// Proprietary UHG Grouping (Primary Dx)
var etg_link = spark.read.format("csv").option("header","true").option("inferSchema","true").load("/user/rrutherford/zeppelin_out/etg_icdx.csv")

// Joined etg_Dx to get some diagnosis dimension reduction. ~100K to ~450
// set custom schema so we get strings instead of ints for etg_base. Helps with vectorizer later.

val etgDxcustomSchema = StructType(Array(
      StructField("PATID", LongType, nullable = true),
      StructField("DIAG", StringType, nullable = true),
      StructField("FST_DT", TimestampType, nullable = true),
      StructField("ETG_BASE", StringType, nullable = true),
      StructField("ICDDX", StringType, nullable = true),
    StructField("ICD_VERSION", IntegerType, nullable = true))
    )
    
val trainingDatcustomSchema = StructType(Array(
      StructField("unnamed", IntegerType, nullable = true),
      StructField("patid", LongType, nullable = true),
      StructField("outcome", DoubleType, nullable = true),
      StructField("testindex", LongType, nullable = true))
    )

var etgDx = spark.read.format("csv").option("header","true").schema(etgDxcustomSchema).load("/user/rrutherford/zeppelin_out/etgDx.csv")

// setting inferSchema to true because I couldn't get the outcome column to cast to integer or float later. 
var trainingDat = spark.read.format("csv").option("header", "true").schema(trainingDatcustomSchema).load("/user/mschlomka/backpain/trainingData.csv").withColumnRenamed("outcome","label")

// MBR Cleaning
mbr = mbr.withColumn("zip_trim", substring(col("ZIPCODE_5"),0,5))
mbr = mbr.withColumn("age",substring(col("ELIGEFF"),0,4)-mbr("YRDOB"))

mbr = mbr.withColumnRenamed("PATID","patid")

// Create categoricals off of Member Table
var features = Array("BUS","GDR_CD","PRODUCT") // drop zip since it leads to feature vec size mis-match train/test.

val encodedFeatures = features.flatMap{ name =>

val stringIndexer = new StringIndexer()
    .setInputCol(name)
    .setOutputCol(name + "_indx")
    
    val oneHotEncoder = new OneHotEncoderEstimator()
    .setInputCols(Array(name + "_indx"))
    .setOutputCols(Array(name + "_vec"))
    .setDropLast(false)
    
    Array(stringIndexer,oneHotEncoder)
}

mbr = new Pipeline().setStages(encodedFeatures).fit(mbr).transform(mbr)


// create Join ETG Link table and Dx.
// Have had issues running in Zeppelin?
// val etgDx = diag.join(etg_link,diag("DIAG")===etg_link("ICDDX"),"inner")

// Create an ETG vector per patient with any diagnosis. We'll later fill everyone who doesn't have a Dx.
val etgDxPivot = etgDx.select("PATID","ETG_BASE").groupBy("PATID").agg(collect_list("ETG_BASE").alias("ETG_BASE_ARRAY")).withColumnRenamed("PATID","patid")

val cv = new CountVectorizer().setInputCol("ETG_BASE_ARRAY").setOutputCol("ETG_BASE_ARRAY" + "_vec")

var train = trainingDat.join(etgDxPivot,Seq("patid"),"left").withColumnRenamed("outcome","label")
val fill = array().cast("array<string>") // Fill pats without Dx to empty array. 
train = train.withColumn("ETG_BASE_ARRAY",coalesce($"ETG_BASE_ARRAY",fill))
train = train.join(mbr,Seq("patid"),"left")

// Create full feature vector from Dx and Mbr
var all_features = Array("age","ETG_BASE_ARRAY_vec") ++ train.columns.filter(_.contains("_vec"))
val vectorAssembler = new VectorAssembler().setInputCols(all_features).setOutputCol("features")
val pipelineVectorAssembler = new Pipeline().setStages(Array(cv,vectorAssembler))
train = pipelineVectorAssembler.fit(train).transform(train)

val lr = new LogisticRegression()
val rf = new RandomForestClassifier().cacheNodeIds(true)
val gbt = new GBTClassifier().cacheNodeIds(true)

val lrPipe = Array[PipelineStage](lr)
val rfPipe = Array[PipelineStage](rf)
val gbtPipe = Array[PipelineStage](gbt)

val pipeline = new Pipeline()

val lrPipe_grid = {new ParamGridBuilder()
  .baseOn(pipeline.stages -> lrPipe)
  .addGrid(lr.maxIter, Array(1,5,10,20,30))
  .addGrid(lr.regParam, Array(0.01,0.1,0.25,0.5,0.75,0.9,1.0))
  .addGrid(lr.elasticNetParam, Array(0.0,0.25,0.5,0.75,1.0))
  .addGrid(lr.fitIntercept, Array(true,false))
  .addGrid(lr.threshold, Array(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9))
  .build()}

val rfPipe_grid = {new ParamGridBuilder()
  .baseOn(pipeline.stages -> rfPipe)
  .addGrid(rf.numTrees, Array(10, 20,30,40,50,75,100,150,200,250))
  .addGrid(rf.maxDepth, Array(0,2,5,10,15,20,25,30,50,100,150,200,250,300))
  .build()}

val gbtPipe_grid = {new ParamGridBuilder()
  .baseOn(pipeline.stages -> gbtPipe)
  .addGrid(gbt.maxIter, Array(5,10,20,50,100,200,300,400))
  .addGrid(gbt.maxDepth,Array( 0,2,5,10,15,20,25,30,50,100,150,200,250,300))
  .build()}
  
val paramGrid = lrPipe_grid ++ rfPipe_grid ++ gbtPipe_grid

// 4 got ~6.8% queue utilization
val cxv = {new CrossValidator() 
  .setEstimator(pipeline)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3)
  .setParallelism(10)}

train.cache().show(5) // cache the training data.

// Run cross-validation, assuming "training" data exists
val cvModel = cxv.fit(train)

// Get the best selected pipeline model
val pipelineModel = cvModel.bestModel.asInstanceOf[PipelineModel]

//

val splits = train.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))

val lrPipeline = new Pipeline().setStages(Array(cv, lr))

val lrModel = lrPipeline.fit(trainingData)

val predictions = lrModel.transform(testData)

// EVALUATE
val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")

val accuracy = evaluator.evaluate(predictions)

val lp = predictions.select( "label", "prediction")
val counttotal = predictions.count()
val correct = lp.filter($"label" === $"prediction").count()
val wrong = lp.filter(not($"label" === $"prediction")).count()
val truep = lp.filter($"prediction" === 0.0).filter($"label" === $"prediction").count()
val falseN = lp.filter($"prediction" === 0.0).filter(not($"label" === $"prediction")).count()
val falseP = lp.filter($"prediction" === 1.0).filter(not($"label" === $"prediction")).count()
val ratioWrong=wrong.toDouble/counttotal.toDouble
val ratioCorrect=correct.toDouble/counttotal.toDouble



// Write out results
val out_df = score_result.select("testindex","prediction").withColumnRenamed("prediction","Predicted")
out_df.repartition(1).write.option("header", "true").csv("/user/rrutherford/zeppelin_out/output_prediction_countVectorizer.csv")
