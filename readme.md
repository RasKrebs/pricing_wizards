# Data Mining for Price Prediction at Trendsales a.k.a. pricing_wizards
## Leveraging Machine Learning Techniques for the Nordics' largest Fashion Marketplace

This is a repository for all programming related assets used in the exam project, Pricing Wizards, for the course Data Mining at ITU.

### Usage

To run code, please place the dataset from the uploaded .zip folder in your local data folder. The background is that due to data being proprietary, we may not upload it to this public GitHub repo.

### Motivation

Given the recommendation of a plain colab link, the format of this deliverable may come as a surprise. As outlined in the Introduction of the accompanying report, the dataset was selected to apply the Data Mining techniques learned in the course in a real-life setting. In this same vein, we collaborated in GitHub and followed other coding best practices, as visible in this folder, like
* Pickling models and storing them in a dedicated folder to enable reproducability
* Isolating function components into dedicated, importable scripts
* Establishing data processing pipelines in a dedicated class
* Operating with a requirements.txt based virtual environment

### Project Methodology

![Alt text](workflow.png)

### Folder Structure

- `/data`: Directory to store dataset
- `/models`: Directory for python model script. These script are referenced when running training, and automize the training process for these
    - `/pickled_models`: Storage for pickled versions of trained models
- `/notebooks`: Used for training of models and data exploration
- `/scripts`: Used to store python scripts
- `/utils`: Helper functions


### Explore Results
If you wish to explore modeling results, please refer to the validation notebook in the root of the repository. This notebook has been created to combine and display all the results of our optimized models. 

To train a model from scratch, based on the best defined preprocessing method, please run:
```ssh
> python train.py --name <one of: base_regression, regularized_regression, neural_network, svm, random_forest>
```

To explore the methodology and process applied for training our models, please refer to the notebooks folder, and open the notebook associated to our models. In here you will also find the preprocessing, exploratory data analysis and outlier detection notebooks, that prepared the data.


### Mapping Methodology to Repo

| Methodology          | Location in repo               |
|----------------------|--------------------------------|
| Data pull            | data_pull.sql                  |
| Basic preprocessing  | notebooks/preprocessing.ipynb  |
| Outlier handling     | utils/Dataloader.py            |
| Encoding             | utils/DataTransformation.py    |
| RQ1: Clustering      | notebooks/clustering.ipynb     |
| RQ2: Price prediction| notebooks, models, utils       |
| RQ3: Feature importance | ADD HERE                    |
