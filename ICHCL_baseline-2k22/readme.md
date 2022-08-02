
## About the problem
Check out the tasks we are offering on our [CFP webpage.]("https://hasocfire.github.io/hasoc/2022/call_for_participation.html") 
If you are interested, [register]("https://hasocfire.github.io/hasoc/2022/registration.html") and join our [mailing list]("https://groups.google.com/g/hasoc") for updates.

## Data
The dataset for this year and previous datasets are available on our data [webpage]("https://hasocfire.github.io/hasoc/2022/dataset.html").

## Baseline
We understand that FIRE hosts so many beginner friendly workshops every year and this problem might not seem like beginner friendly. So, we’ve decided to provide participants with a baseline model which will provide participants with a template for steps like importing data, preprocessing, featuring and classification. And the participants can make changes in the code and experiment with various settings. 

Note: baseline model is just to give you a basic idea of our dir. structure and how one can classify context based data, there are no restrictions on any kind of experiments

## Requirements
```
pip install -r requirements.txt
```


## Run the code
To run the model for binary task
```
python binary.py <path to data folder>
```

Similarly, to run the model for mutli class task
```
python multiclass.py <path to data folder>
```

