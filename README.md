# 🧠 Melanoma Image Classification & Synthetic Data Generation using Deep Learning & GANs


This repository documents the development and implementation of a deep learning pipeline designed to support the automated classification of melanoma from dermoscopic images. The project integrates a custom-designed Convolutional Neural Network (CNN) with a Conditional Wasserstein Generative Adversarial Network with Gradient Penalty (cWGAN-GP). It contributes to the growing field of AI-driven medical diagnostics by aiming to accurately distinguish between benign and malignant skin lesions, reduce the reliance on manual labelling, and strategically use generative AI to improve the quality and balance of training data. By enhancing both the accuracy and sensitivity of melanoma detection, the system supports earlier diagnosis and reduces the diagnostic burden on healthcare professionals. This report outlines the full development lifecycle from data acquisition and model design to evaluation and discusses the broader implications for medical image analysis and clinical practice.

## 📌 Project Objectives

The main objective of this project was to design and implement a deep learning framework capable of accurately classifying skin lesions as either benign or malignant (melanoma). A key challenge in training robust classification models for melanoma is the limited availability of malignant samples compared to benign ones. This imbalance can lead to biased learning, where the model favors the majority class. To mitigate this, generative artificial intelligence was employed to artificially increase the number of melanoma examples, thereby creating a more balanced and diverse dataset. Moreover, the project explored the broader ethical and clinical implications of AI-based diagnostic tools, particularly in terms of their reliability, interpretability, and potential use in real-world clinical settings.


## 🛠️ Tools & Technologies

The project was developed using Python 3.9 and leveraged several key machine learning and data science libraries including TensorFlow 2.13 with Keras, Pandas, NumPy, and Matplotlib. Early experimentation was conducted in Jupyter Notebooks, with the bulk of model training performed in Google Colab due to its access to GPU resources, specifically an NVIDIA Tesla T4 with 16GB of RAM. The conditional GAN was implemented to enhance the training data, while a custom CNN was built from scratch to ensure transparency and interpretability in the classification process. The combination of these tools allowed for efficient handling of large-scale image data, robust training, and detailed performance analysis.

## 📊 Datasets

The primary dataset used for this study was the SIIM-ISIC 2020 Melanoma Classification dataset, available on Kaggle. It contains over 33,000 dermoscopic images, but only approximately 1.7% of them are labeled as melanoma, presenting a significant class imbalance challenge. This dataset was selected due to its clinical relevance, metadata richness, and scale. It provided a reliable foundation for training and evaluating the deep learning models. The project also considered the potential integration of additional datasets such as HAM10000, PH2, and Fitzpatrick17k, to increase diversity and support future development.


![image](https://github.com/user-attachments/assets/82b5c5b5-f0c6-4690-85ec-1a48a51d79b4)



## ⚙️ Methodology

The methodology employed in this project followed a structured software development life cycle with Agile principles. Initially, a comprehensive preprocessing pipeline was developed to ensure consistency in image dimensions and quality. All images were resized to 224x224 for CNN training and normalized appropriately. The cWGAN-GP was trained using only melanoma images to generate synthetic samples that reflected the minority class. This training involved feeding noise vectors and labels to the generator and training the critic (discriminator) to distinguish between real and generated images. After generating 1,000 synthetic images, they were integrated into the training set. A custom CNN classifier was trained twice: first on the original data, and again on the GAN-augmented dataset. Performance was evaluated using accuracy, precision, recall, and AUC, and further analyzed through ROC curves and training histories.

![image](https://github.com/user-attachments/assets/b61dbc04-cf4a-4e7c-a8ae-a9b4aac6670d)



## 📈 Key Findings

The key findings of this project demonstrate that using GAN-based data augmentation can significantly increase a model’s sensitivity to detecting melanoma cases. The recall rate improved from around 12% in the baseline model to 100% in the GAN-augmented model. However, this gain in sensitivity came at the cost of specificity and overall accuracy, leading to a rise in false positives. This tradeoff highlights the classic challenge in medical AI of balancing false negatives and false positives. While the GAN augmentation improved the model's ability to detect rare malignant cases, it also caused the model to misclassify some benign lesions. These findings underscore the need for further calibration and validation of such systems before clinical deployment.

![image](https://github.com/user-attachments/assets/5750c943-fdb3-4b2f-9ce4-2eb65a5a85bc)


![image](https://github.com/user-attachments/assets/44116a4e-70ce-4cbe-bfc0-50ab51c9541c)

![image](https://github.com/user-attachments/assets/8de24a54-b50e-4800-810b-92e8f99a98fa)



## ❗ Limitations

Despite its promising results, the project faced several limitations. The number of real melanoma images available for GAN training was small (only 584 samples), which may have limited the diversity of generated images. This constraint increases the risk of mode collapse, where the generator produces repetitive or overly similar images. Additionally, the synthetic images, while visually convincing, were not validated against clinical diagnostic criteria, such as the ABCD rule or dermatologist review, and may not represent real-world pathological patterns. The project was also constrained by computational limits of the free Google Colab environment, which imposed training timeouts and limited memory availability, potentially affecting model optimization.

## 🔮 Future Work

Several opportunities exist to enhance and expand this work. Incorporating more diverse and extensive datasets would improve both the GAN and the classifier’s generalization abilities. A formal validation process, potentially involving dermatologists and metrics like the Fréchet Inception Distance (FID), should be introduced to assess the clinical realism of synthetic images. Image preprocessing can be improved through techniques like automated hair removal and color normalization. Exploring more powerful generative models such as StyleGAN2, and using transfer learning with advanced classifiers like EfficientNet or Vision Transformers, could enhance model performance. Finally, future efforts should evaluate the models in real-world clinical settings using human-in-the-loop systems to understand their practical utility and impact on patient care.



## 🙋‍♂️ Author

**Ismail Dahir**  
