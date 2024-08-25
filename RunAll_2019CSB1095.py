import sys
import os
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

# get mnist-fashion dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 0 T-shirt/top     # 5 Sandal
# 1 Trouser         # 6 Shirt
# 2 Pullover        # 7 Sneaker
# 3 Dress           # 8 Bag
# 4 Coat            # 9 Ankle boot

# set folder path
folder_path = "./"
  
demo = False

# get distance between 2 lists (taken as a metric for evaluation)
def getDistance(list1, list2):
  # using root mean square distance
  dist = 0
  for i in range(len(list1)):
    dist += (list1[i]-list2[i])**2
  dist /= len(list1)
  return np.sqrt(dist)

# fucntion to perform k-means clustering
def KMeans(k:int, desc_list:np.ndarray, num_iterations:int = 6):

  # randomly initialize centroids
  random_indices = random.sample(range(1, len(desc_list)), k)

  centroids = [desc_list[k-1] for k in random_indices]

  centroid_id = [-1 for _ in desc_list]

  for iter in range(num_iterations):

    if demo:
      print(f"iteration {iter+1} started...")

    # asssign centroids
    for j in range(len(desc_list)):
      desc = desc_list[j]
      min_dist = np.inf
      curr_centroid = -1
      for i in range(len(centroids)):
        curr_dist = np.sqrt(np.sum(((desc-centroids[i])**2)/len(desc)))
        if curr_dist < min_dist:
          min_dist = curr_dist
          curr_centroid = i
      centroid_id[j] = curr_centroid

    if demo:
      print("first loop done...")

    new_centroids = np.array([[0 for i in range(len(j))] for j in centroids])
    cluster_size = np.array([0 for i in centroids])

    for i in range(len(desc_list)):
      new_centroids[centroid_id[i]] += desc_list[i]
      cluster_size[centroid_id[i]] += 1

    # recalculate centroids
    for i in range(len(new_centroids)):
      if cluster_size[i] == 0:
        new_centroids[i] = desc_list[random.randint(1,len(desc_list))-1]
      else:
        new_centroids[i] = (new_centroids[i]/cluster_size[i]).astype("int")
    
    if demo:
      print("new centroids calculated!")
    
    # check if there is very less change
    change = 0
    for i in range(len(centroids)):
      change += getDistance(centroids[i], new_centroids[i])
    if change < 400:
      if demo:
        print("stable at iteration", iter+1)
      break

    centroids = new_centroids

  return centroids




def ComputeHistogram(img:np.ndarray, centroids):
  hist = [0 for _ in range(len(centroids))]
  for y in range(0, 28, 7):
    for x in range(0, 28, 7):
      block = img[y:y+7, x:x+7].flatten()
      min_dist = np.inf
      curr_centroid = -1
      for i in range(len(centroids)):
        curr_dist = np.sqrt(np.sum(((block-centroids[i])**2)/len(block)))
        if curr_dist < min_dist:
          min_dist = curr_dist
          curr_centroid = i
      hist[curr_centroid] += 1
  return hist


def MatchHistogram(hist1, hist2):
  dist = 0
  for i in range(len(hist1)):
    dist += (hist1[i]-hist2[i])**2
  dist /= len(hist1)
  return np.sqrt(dist)

def ClosestHistogram(img:np.ndarray, histograms, final_centroids):

  hist = ComputeHistogram(img, final_centroids)

  min_dist = np.inf
  curr_hist = -1
  for i in range(len(histograms)):
    # curr_dist = np.sqrt(np.sum(((np.array(hist)-np.array(histograms[i]))**2)/len(hist)))
    curr_dist = MatchHistogram(hist, histograms[i])
    if curr_dist < min_dist:
      min_dist = curr_dist
      curr_hist = i
  
  return curr_hist
  

def CreateVisualDictionary(all_desc):
    
  if demo:
    print("k means started")

  final_centroids = KMeans(100, all_desc)

  if not os.path.exists(folder_path+'visual_words'):
    os.mkdir(folder_path+'visual_words') 
  img_num = 1
  for centroid in final_centroids:
    img = centroid.reshape((7,7))
    cv2.imwrite(folder_path+"visual_words/centroid_{}.png".format(img_num), img)  # Save image localy 
    img_num += 1

  # write to file in case of data loss due to error
  file_name = "kmeans.txt" if not demo else "kmeans_demo.txt"
  f = open(folder_path+file_name, "w")
  for k in final_centroids:
    f.write(" ".join(map(str,k)))
    f.write("\n")
  f.close()

  return final_centroids

# calculate required metrics
def CalculateMetrics(results):
  data = {}
  data["truePos"] = [0 for _ in range(10)]
  data["falsePos"] = [0 for _ in range(10)]
  data["falseNeg"] = [0 for _ in range(10)]
  data["trueNeg"] = [0 for _ in range(10)]

  total = len(results["predicted_label"])

  for i in range(total):
    correct = results["actual_label"][i]
    predicted = results["predicted_label"][i]
    if correct == predicted:
      data["truePos"][correct] += 1
    else:
      data["falseNeg"][correct] += 1
      data["falsePos"][predicted] += 1

  for i in range(10):
    data["trueNeg"][i] = total - (data["truePos"][i] + data["falseNeg"][i] + data["falsePos"][i])
  
  total_correct = np.sum(data["truePos"])

  overall_accuracy = (100*total_correct)/total

  print(f"Overall classification accuracy = {overall_accuracy}%")

  print("Class\tAccuracy\tPrecision\tRecall")

  for i in range(10):

    accuracy = 100*(data["truePos"][i])/(data["truePos"][i]+data["falseNeg"][i])
    precision = data["truePos"][i]/(data["truePos"][i]+data["falsePos"][i])
    recall = data["truePos"][i]/(data["truePos"][i]+data["falseNeg"][i])

    print("{}\t{:.3f} %\t{:.3f}\t\t{:.3f}".format(i, accuracy, precision, recall))

  return data
  

def main(args):
  
  global demo
  global x_train

  if len(args) > 1:
    if args[1] == "--demo":
      demo = True
  
  if demo:
    x_train = x_train[:200]


  # extract feature descriptors from each image
  all_desc = []

  for i in range(len(x_train)):
    for y in range(0, 28, 7):
      for x in range(0, 28, 7):
        sub_img = x_train[i][y:y+7, x:x+7]
        desc = sub_img.flatten()
        desc_roundoff = ((desc//5)*5).astype("int")
        all_desc.append(np.append(desc_roundoff, y_train[i]))

  # write to file to avoid data loss in case of error
  file_name = "descriptors.txt" if not demo else "descriptors_demo.txt"
  f = open(folder_path+file_name, "w")
  for desc in all_desc:
    f.write(" ".join(map(str,desc)))
    f.write("\n")
  f.close()


  for i in range(len(all_desc)):
    all_desc[i] = all_desc[i][:-1]

  # get centroids using k means
  final_centroids = CreateVisualDictionary(all_desc)

  # calculate histograms for training data
  histogram_train = [None for _ in range(len(x_train))]

  for i in range(len(x_train)):
    histogram_train[i] = ComputeHistogram(np.array(x_train[i]), final_centroids)
    if i%100 == 0:
      print(i, end = " ")
      if i%10000 == 0:
        print()

  # save data to file
  file_name = "histograms.txt" if not demo else "histograms_demo.txt"
  f = open(folder_path+file_name, "w")
  for k in histogram_train:
    if not k:
      continue
    f.write(" ".join(map(str,k)))
    f.write("\n")
  f.close()

  # calculate results for test data
  results = {"id":[], "closest_hist":[], "actual_label":[], "predicted_label":[]}

  correct = 0
  incorrect = 0

  for i in range(len(x_test)-9900*demo):
    
    closest_id = ClosestHistogram(np.array(x_test[i]), histogram_train, final_centroids)

    results["id"].append(i)
    results["closest_hist"].append(closest_id)
    results["actual_label"].append(y_test[i])
    results["predicted_label"].append(y_train[closest_id])

    if demo:
      print(f'{results["id"][i]} {results["closest_hist"][i]} {results["actual_label"][i]} {results["predicted_label"][i]}')

    if y_test[i] == y_train[closest_id]:
      correct+=1
    else:
      incorrect+=1
    

  # write data to file
  file_name = f"results.txt" if not demo else "results_demo.txt"
  f = open(folder_path+file_name, "w")
  for i in range(len(results["id"])):
    f.write(f'{results["id"][i]} {results["closest_hist"][i]} {results["actual_label"][i]} {results["predicted_label"][i]}\n')
  f.close()

  # get required metrics from results
  CalculateMetrics(results)

  return


# run main
if __name__ == "__main__":
  main(sys.argv)


