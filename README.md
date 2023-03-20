### Colorinsight (Personal Color detection model, 퍼스널컬러 진단모델) <br>
https://user-images.githubusercontent.com/86555104/226335673-e7cb3db0-7128-4fcb-9c9e-3c397ecd22f1.mp4
<br> Personal color refers to the colors that look best on an individual based on their skin tone, hair color, and eye color. Personal color analysis is a process that helps identify an individual's personal color palette, which consists of a range of colors that complement their natural features. This concept is used in the fashion and beauty industries to help individuals choose colors for clothing, makeup, and accessories that flatter their appearance. <br>

Our team built the website called 'Colorinsight' because of two big motivations: 1) Current private personal color consulting services have low reliability due to subjective diagnosis compared to high prices. 2) The demand for personalized beauty industry is growing worldwide. Our model is developed based on a thorough research on the existing personal color theories and concrete experiments on deep learning models to make practical service for the users.

### ⛏ Model Overview

![image](https://user-images.githubusercontent.com/86555104/226334045-07eddda5-61ac-4446-9bc8-07edeb7090f2.png)
We implemeted two separated models for the personal color diagnosis. After the careful comparison of performance between several models, we selected FaRL model which showed outstanding performance compared to other face segmentation models even in complicated situation(face in different direction, extreme face shape...etc.). After sementing pure skin part of face, we trained another image classification model with Korean celebrity images. The dataset was collected by Google image crawling using python selenium. To overcome the limited image data(750 images), data augmentation was employed. (dataset is not uploaded on a repository for a privacy protection)


### A. Deel Learning Model Backbone (FaRL model)
The backbone of this personal color detection model is based on a face parsing model from FaRL(Facial Representation Learning) from the great authors below.

@article{zheng2021farl,
  title={General Facial Representation Learning in a Visual-Linguistic Manner},
  author={Zheng, Yinglin and Yang, Hao and Zhang, Ting and Bao, Jianmin and Chen, Dongdong and Huang, Yangyu and Yuan, Lu and Chen, Dong and Zeng, Ming and Wen, Fang},
  journal={arXiv preprint arXiv:2112.03109},
  year={2021}
}<br>

### C. Personal Color based on Face Skin diagnosis model
After obtaining face skin mask image, we tried three approaches to find the most suitable model that matches an exact personal color type. (Personal color type is divided into four categories: spring, summer, autumnm, winter) First approach was to categorize personal color type with L2 norm distance between particular rgb code data point from the research paper and randomly extracted rgb codes from skin mask area. For the second approach, we made additional structured dataset with r,g,b columns extracted from face skin mask. The, we applied machine learning classification model to predict the proper personal color type label. However, these two methods resulted low accuracy rate ranging from 20% to 30 % with the limitation that it fails to predict 'autumn' type. <br>

As a last attempt , we organized another image dataset that extracted face skin mask image from previously collected Korean celebrity face dataset. Then we tested popular image classification models including MobileNet, ResNet and EfficientNet. ResNet with Adam optimizer demonstrated the best performance among all, that we eventually used this model for the prediction. 
*ResNet was trained in Colab and loaded model via best_model_resnet_ALL.pth file.
![image](https://user-images.githubusercontent.com/86555104/226340188-2cfe2cba-23f5-4112-a51f-cfe9e1c32b12.png)


### Possible Improvements
- It is successful that the accuracy of model is improved from 20% to around 60%, however, additional efforts such as increased training epoch and dataset expansion can improve model's performance.
- Current dataset is based on Korean celebrity photos, but utilizing photos of ordinary people in different angles will enable model to learn realistic human skin color and expand its usage on diverse races.<br>

Detailed experiment reports can be found here(written in Korean) 👉[Deep Learning Model Experiment Reports](https://tar-tilapia-c6d.notion.site/403c8d583e3a4f6bb9f76ea6efd991d5?v=f9b650bea3e144918ec577eb464ddcd5/)
