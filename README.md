# bow_based_categorization_on_mnist
Bag of words based matching/categorization solution on the MNIST-fashion database.
 
Data downloaded from- 
https://github.com/zalandoresearch/fashion-mnist/blob/master/README.md 
https://www.kaggle.com/zalando-research/fashionmnist/data 

### Functions Implemented -

* **CreateVisualDictionary()** – computes and save the visual dictionary (K-means implemented manually)

* **ComputeHistogram()** – takes as input a feature vector and the visual dictionary matrix and generates the histogram using soft assignment (giving weight to the next nearest neighbor)

* **MatchHistogram()** – the function compares two histograms and returns the distance.

Entry script **RunAll_2019csb1095.py** extracts features from the images and then calls the CreateVisualDictionary() function. 
Then the script calls the ComputeHistogram() function to create the histograms for all the training and Test images. 
MatchHistogram() then is called for the Test set images and the Label (category) for each Test image is generated by assigning the class of the nearest neighbor (in the Training set.
 
The overall classification accuracy, class wise accuracy, precision and recall are displayed.

Refer **Report.docx** for further details.
