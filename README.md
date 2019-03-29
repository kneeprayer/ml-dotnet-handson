# ML.NET Tutorial
This tutorial is based on [Microsort ML.Net Official Document](https://dotnet.microsoft.com/learn/machinelearning-ai/ml-dotnet-get-started-tutorial/intro)

# Download and install on Mac
1. Install .Net core SDK using Homebrew
    ```
    brew cask install dotnet-sdk
    ```
2. Check everything installed correctly.
    ```
    dotnet
    ```
3. Check your .Net Core SDK version.
    ```
    dotnet --version
    ```
4. Install C# Extensions on vscode.
    [C# for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=ms-vscode.csharp)

# Create your app
1. In your terminal, run the following commands:
    ```
    dotnet new console -o myMLApp
    cd myMLApp
    ```
2. Install ML.NET package
    ```
    dotnet add package Microsoft.ML
    ```
# Download data set
Your machine learning app will predict the type of iris flower (setosa, versicolor, or virginica) based on four features: petal length, petal width, sepal length, and sepal width

Open the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data): Iris Data Set, copy and paste the data into a text editor (e.g. Notepad), and save it as iris-data.txt in the myMLApp directory.

When you paste the data it will look like the following. Each row represents a different sample of an iris flower. From left to right, the columns represent: sepal length, sepal width, petal length, petal width, and type of iris flower.

# Write some code
Open Program.cs in any text editor and replace all of the code with the following:
```
    using Microsoft.Data.DataView;
    using Microsoft.ML;
    using Microsoft.ML.Data;
    using System;
    // CS0649 compiler warning is disabled because some fields are only
    // assigned to dynamically by ML.NET at runtime
    #pragma warning disable CS0649
    
    namespace myMLApp
    {
        class Program
        {
            // STEP 1: Define your data structures
            // IrisData is used to provide training data, and as
            // input for prediction operations
            // - First 4 properties are inputs/features used to predict the label
            // - Label is what you are predicting, and is only set when training
            public class IrisData
            {
                [LoadColumn(0)]
                public float SepalLength;

                [LoadColumn(1)]
                public float SepalWidth;

                [LoadColumn(2)]
                public float PetalLength;

                [LoadColumn(3)]
                public float PetalWidth;

                [LoadColumn(4)]
                public string Label;
            }

            // IrisPrediction is the result returned from prediction operations
            public class IrisPrediction
            {
                [ColumnName("PredictedLabel")]
                public string PredictedLabels;
            }

            static void Main(string[] args)
            {
                // STEP 2: Create a ML.NET environment
                MLContext mlContext = new MLContext();

                // If working in Visual Studio, make sure the 'Copy to Output Directory'
                // property of iris-data.txt is set to 'Copy always'
                IDataView trainingDataView = mlContext.Data.LoadFromTextFile<IrisData>(path: "iris-data.txt", hasHeader: false, separatorChar: ',');

                // STEP 3: Transform your data and add a learner
                // Assign numeric values to text in the "Label" column, because only
                // numbers can be processed during model training.
                // Add a learning algorithm to the pipeline. e.g.(What type of iris is this?)
                // Convert the Label back into original text (after converting to number in step 3)
                var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                    .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                    .AppendCacheCheckpoint(mlContext)
                    .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumnName: "Label", featureColumnName: "Features"))
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                // STEP 4: Train your model based on the data set
                var model = pipeline.Fit(trainingDataView);

                // STEP 5: Use your model to make a prediction
                // You can change these numbers to test different predictions
                var prediction = model.CreatePredictionEngine<IrisData, IrisPrediction>(mlContext).Predict(
                    new IrisData()
                    {
                        SepalLength = 3.3f,
                        SepalWidth = 1.6f,
                        PetalLength = 0.2f,
                        PetalWidth = 5.1f,
                    });

                Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");

                Console.WriteLine("Press any key to exit....");
                Console.ReadLine();
            }
        }
    }
```

# Run your app
In your terminal, run the following command:
    ```
    dotnet run
    ```

# Keep learning
1. Now that you've got the basics, you can keep learning with our ML.NET tutorials.
    [.NET Machine learning tutorials - ML.NET](https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/)
2. You might also be interested in...
    [ML.NET Samples](https://github.com/dotnet/machinelearning-samples/blob/master/README.md)