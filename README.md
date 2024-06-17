# Comparative-Analysis-of-Generative-Adversarial-Network
# ABSTRACT

This project explores the capabilities of adversarial training and cycle consistency constraints in unpaired image-to-image translation tasks, focusing on transforming images between two distinct domains: horses and zebras. Inspired by the success of CycleGAN, we aim to demonstrate the potential for realistic image transformations across diverse visual domains. Furthermore, leveraging the ChestX-ray dataset, which includes labeled images categorized as "NORMAL" and "PNEUMONIA," also we have Vanilla GAN which is implemented on brain MRI images for brain tumor detection dataset .We extend our investigation to the realm of medical imaging. Here, we employ an Auxiliary Classifier Generative Adversarial Network (ACGAN) architecture to generate chest X-ray images, offering insights into potential applications in medical image synthesis. The generator model architecture comprises multiple convolutional layers, supplemented by transpose convolutional layers. These layers work together to up sample the input noise vector, ultimately producing high-fidelity and realistic images representative of the target domain.
Through this research, we showcase the efficacy of adversarial training techniques and cycle consistency constraints not only in artistic image synthesis but also in the generation of medical images, thereby highlighting the versatility and potential impact of such approaches across various domains.

# INTRODUCTION
A Generative Adversarial Network (GAN) is a type of artificial intelligence algorithm composed of two neural networks, the generator and the discriminator, which are trained together in a competitive manner. The generator generates synthetic data, such as images, while the discriminator evaluates the authenticity of the generated data. Through iterative training, the generator learns to produce increasingly realistic data, while the discriminator becomes more adept at distinguishing between real and fake data. GANs have demonstrated remarkable success in generating high-quality synthetic data across various domains, including images, audio, and text. Generative Adversarial Networks (GANs) operate on a fundamentally adversarial principle, comprising two neural networks—the generator and the discriminator—that engage in a competitive learning process. The generator is tasked with synthesizing data, such as images, from random noise, aiming to produce samples that are indistinguishable from real data. On the other hand, the discriminator acts as a binary classifier, distinguishing between real and fake data. During training, the generator generates fake samples, and the discriminator evaluates their authenticity. The discriminator provides feedback to the generator, guiding it to produce more realistic samples. Simultaneously, the discriminator updates its parameters to become more proficient at distinguishing real from fake data. This adversarial interplay continues iteratively, with both networks improving over time through a process akin to a minimax game, where the generator aims to minimize the discriminator's ability to differentiate between real and fake data while the discriminator strives to accurately classify the samples. As training progresses, the generator learns to generate increasingly realistic samples, converging towards a distribution that closely resembles the true data distribution. The ultimate goal of a GAN is to produce synthetic data that is indistinguishable from real data, capturing the underlying patterns and structure of the dataset it was trained on. This process enables GANs to generate high-quality synthetic data across various domains, facilitating tasks such as image synthesis, data augmentation, and anomaly detection.

![image](https://github.com/Himanshu300703/Comparative-Analysis-of-Generative-Adversarial-Network/assets/91286198/b3783cd7-fdc4-4a3c-972e-1bd328da4bc1)

Fig: Working architecture of GAN

In the realm of artificial intelligence and computer vision, the ability to translate images between different visual domains has garnered significant attention and acclaim. This endeavour, known as image-to-image translation, holds promise for a myriad of applications, ranging from artistic expression to medical imaging. One prominent approach in this domain is the CycleGAN framework, which exemplifies the power of adversarial training and cycle consistency constraints in achieving realistic transformations between disparate visual domains. 
The core idea behind CycleGAN lies in its ability to learn mappings between two domains without the need for paired data, thereby obviating the laborious task of collecting matched image pairs for training. Instead, it leverages unpaired images from each domain and employs adversarial training to learn the mapping functions implicitly. This methodology has been demonstrated to yield impressive results, enabling the creation of compelling transformations between diverse domains such as horses and zebras. Beyond artistic endeavours, the potential applications of image-to-image translation extend into the realm of medical imaging, where the ability to generate realistic synthetic images holds immense value. In this context, the ChestX-ray8 dataset provides a rich source of labelled chest X-ray images, offering an opportunity to explore the synthesis of medical imagery using advanced deep learning techniques. Leveraging this dataset, our project delves into the synthesis of chest X-ray images, aiming to generate both "NORMAL" and "PNEUMONIA" labelled images for diagnostic purposes. 

To accomplish this task, we employ an Auxiliary Classifier Generative Adversarial Network (ACGAN) architecture, which extends the traditional GAN framework by incorporating an auxiliary classifier into the discriminator. This addition facilitates not only the generation of realistic images but also the control over specific attributes or classes within the generated samples. By incorporating class information into the discriminator, the ACGAN framework enables more precise control over the image generation process, ensuring that the synthesized images adhere to desired characteristics. Central to our approach is the generator model, which is tasked with translating noise vectors into realistic images. This model architecture comprises several convolutional layers followed by transpose convolutional layers, facilitating the upsampling of the input noise vector to generate high-resolution images. Inspired by the success of CycleGAN, our model architecture is built upon similar principles, with two generator networks (one for each domain) and two discriminator networks.
The generators are responsible for translating images from one domain to the other, while the discriminators differentiate between real and generated images, providing feedback to the generators to improve their performance iteratively. To enhance the stability and effectiveness of training, we employ UNet-based architectures for both generators and discriminators, leveraging the inherent skip connections to facilitate information flow between different layers. Additionally, instance normalization is utilized to stabilize the training process, ensuring consistent and reliable convergence of the model.
In summary, our project endeavours to harness the capabilities of advanced deep learning architectures, namely ACGAN and CycleGAN, for the purpose of unpaired image-to-image translation. By extending these frameworks to the domain of medical imaging, we aim to demonstrate their efficacy in synthesizing realistic chest X-ray images, thereby potentially contributing to advancements in diagnostic imaging and healthcare.

# RESULTS 
GAN’s	FID SCORE

Vanilla Gan - 40.845

ACGAN	- 15.2875

CycleGAN - 34.561

# ACGAN Results

![image](https://github.com/Himanshu300703/Comparative-Analysis-of-Generative-Adversarial-Network/assets/91286198/96a65044-ecc7-48e9-81eb-bcc9ecaaa82a)

Fig: Transformed with ACGAN

![image](https://github.com/Himanshu300703/Comparative-Analysis-of-Generative-Adversarial-Network/assets/91286198/a54aa3a0-1dfb-41f4-a059-6077716dbfad)

Fig: Train and Validation loss of ACGAN 

# Cycle GAN Results

![image](https://github.com/Himanshu300703/Comparative-Analysis-of-Generative-Adversarial-Network/assets/91286198/9ec8d9f1-f86d-4ee2-9744-af198cc5561c)

Fig: Image with random jitter

![image](https://github.com/Himanshu300703/Comparative-Analysis-of-Generative-Adversarial-Network/assets/91286198/cb97026f-8aa3-42df-a8ab-0fa435f19dc2)

Fig: Transformed with CycleGAN

# Vanilla GAN Results

![image](https://github.com/Himanshu300703/Comparative-Analysis-of-Generative-Adversarial-Network/assets/91286198/1b7fb966-56ac-42c4-9878-86128b8ae96a)

Fig: Vanilla GAN generated brain MRI image

![image](https://github.com/Himanshu300703/Comparative-Analysis-of-Generative-Adversarial-Network/assets/91286198/ba9ba54b-8dda-4d35-a4ed-9af2f99f39a2)

Fig: Comparing the generated images with the real samples by plotting their distributions. If the distributions overlap, that indicates the generated samples are very close to the real ones

# CONCLUSION

In this paper, Cycle GAN framework showcases remarkable capabilities in unpaired image-to-image translation, exemplified by its successful translation of images between the horse and zebra domains without the requirement for paired training data. Through qualitative evaluation, the model demonstrates its ability to generate visually plausible translations, capturing essential visual characteristics and producing compelling transformations. This underscores Cycle GAN's efficacy in learning domain mappings and generating realistic images, thus presenting a valuable tool for diverse applications, from artistic expression to domain adaptation in computer vision tasks.
Similarly, the ACGAN architecture proves to be a formidable approach for generating chest X-ray images, leveraging adversarial training to synthesize realistic medical imagery. With its auxiliary classifier enhancing control over generated images' attributes, ACGAN effectively produces chest X-ray images with discernible class labels (NORMAL or PNEUMONIA). Through rigorous training and evaluation, ACGAN showcases its potential in medical image synthesis, offering a pathway towards generating high-fidelity medical images for diagnostic and research purposes. Both CycleGAN and ACGAN represent significant advancements in the field of generative adversarial networks, promising avenues for further exploration and application in various domains.
