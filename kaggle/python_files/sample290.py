#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# I love this competition! But the title **"Pneumonia Detection"** for the competition is misleading because you actually have to do **"Lung Opacities Detection"**, and lung opacities are not the same as pneumonia. Lung opacities are vague, fuzzy clouds of white in the darkness of the lungs, which makes detecting them a real challenge.
# 
# This kernel is for people who want to understand what lung opacities are and how come there are "No Lung Opacity / Not Normal" images. In this kernel I will show images from different classes (Normal; No Lung Opacity / Not Normal; Lung Opacity) and write my interpretation of the **opacities** in the image. I will cover some basics of chest radiography and focus only on the lungs because "Lung Opacities" appear only in the lungs. Write your questions in the comments section and I'll answer them as best that I can.
# 
# My name is Guy Zahavi, I am a physician (anesthesiology resident) and graduate bioinformatics student. I am not affiliated with the competition hosts or Kaggle, I'm just excited about this competition :)
# 
# *I took the code for this kernel from Peter Chang's awesome [Exploratory Data Analysis](https://www.kaggle.com/peterchang77/exploratory-data-analysis). Thank you Peter!*

# ## Table of Contents
# 
# - [What Does a Normal Image Look Like?](#What-Does-a-Normal-Image-Look-Like?)
# - [Chest Radiographs Basics](#Chest-Radiographs-Basics)
# - [What are Lung Opacities?](#What-are-Lung-Opacities?)
#     - [The Definition of Opacity](#The-Definition-of-Opacity)
#     - [What are opacities and what can we understand if we see them?](#What-are-opacities-and-what-can-we-understand-if-we-see-them?)
#     - [Normal vs. Lung Opacity pictures](#Normal-vs.-Lung-Opacity-pictures)
# - [Some considerations when building your model](#Some-considerations-when-building-your-model)
#     - [Localization vs. Segmentation of Opacities](#Localization-vs.-Segmentation-of-Opacities)
#     - [How accurate are the annotated boxes?](#How-accurate-are-the-annotated-boxes?)
# - [A Closer Look Into "No Lung Opacity / Not Normal" Images](#A-Closer-Look-Into-"No-Lung-Opacity-/-Not-Normal"-Images)
#     - [Why are there multiple opacities in Not Normal / No Lung Opacities images?](#Why-are-there-multiple-opacities-in-Not-Normal-/-No-Lung-Opacities-images?)
#     - [What makes pneumonia associated opacities unique?](#What-makes-pneumonia-associated-opacities-unique?)
#     - [Lung opacity annotation instructions to radiologists for this dataset](#Lung-opacity-annotation-instructions-to-radiologists-for-this-dataset)
# - [A Clear and Detailed Definition of Pneumonia Associated Lung Opacities](#A-Clear-and-Detailed-Definition-of-Pneumonia-Associated-Lung-Opacities)
#     - [Consolidation vs. Ground-Glass Opacity](#Consolidation-vs.-Ground-Glass-Opacity)
# - [Building Your Model for Pneumonia Associated Lung Opacities](#Building-Your-Model-for-Pneumonia-Associated-Lung-Opacities)
#     - [Are there pneumonia images in the No Lung Opacity / Not Normal class?](#Are-there-pneumonia-images-in-the-No-Lung-Opacity-/-Not-Normal-class?)
#     - [How can I predict if the patient has pneumonia?](#How-can-I-predict-if-the-patient-has-pneumonia?)
# - [Opacities That Are Not Related to Pneumonia](#Opacities-That-Are-Not-Related-to-Pneumonia)
#     - [Pleural effusion](#Pleural-effusion)
#     - [Lung Masses and Nodules](#Lung-Masses-and-Nodules)
#     - [Increased Vascular Markings + Enlarged Heart](#Increased-Vascular-Markings-+-Enlarged-Heart)
#     - [White Lung](#White-Lung)
#     - [Unclear Abnormality](#Unclear-Abnormality)
# - [Summary](#Summary)
# 

# ## Loading and preparing the data

# In[ ]:


import glob, pandas as pd
import matplotlib.pyplot as plt
import pydicom, numpy as np

def parse_data(df):
    """
    Method to read a CSV file (Pandas dataframe) and parse the 
    data into the following nested dictionary:

      parsed = {
        
        'patientId-00': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        },
        'patientId-01': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia, 
            'boxes': list of box(es)
        }, ...

      }

    """
    # --- Define lambda to extract coords in list [y, x, height, width]
    extract_box = lambda row: [row['y'], row['x'], row['height'], row['width']]

    parsed = {}
    for n, row in df.iterrows():
        # --- Initialize patient entry into parsed 
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': '../input/stage_1_train_images/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # --- Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed

df = pd.read_csv('../input/stage_1_train_labels.csv')

patient_class = pd.read_csv('../input/stage_1_detailed_class_info.csv', index_col=0)

parsed = parse_data(df)

patientId = df['patientId'][0]
print('Just a checking that everything is working fine...')
print(parsed[patientId])
print(patient_class.loc[patientId])

def draw(data):
    """
    Method to draw single patient with bounding box(es) if present 

    """
    # --- Open DICOM file
    d = pydicom.read_file(data['dicom'])
    im = d.pixel_array

    # --- Convert from single-channel grayscale to 3-channel RGB
    im = np.stack([im] * 3, axis=2)

    # --- Add boxes with random color if present
    for box in data['boxes']:
        rgb = np.floor(np.random.rand(3) * 256).astype('int')
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)

    plt.imshow(im, cmap=plt.cm.gist_gray)
    plt.axis('off')

def overlay_box(im, box, rgb, stroke=1):
    """
    Method to overlay single box on image

    """
    # --- Convert coordinates to integers
    box = [int(b) for b in box]
    
    # --- Extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im

# # What Does a Normal Image Look Like?
# [Back to top](#Table-of-Contents)
# 
# This is an illustration of the chest anatomy with the lungs highlighted - 
# 
# ![Chest Anatomy](https://i.imgur.com/Jb9OmxM.jpg)
# *Credit: MDGRPHCS / Shutterstock.com*
# 
# You can see that there is a mass of tissue surrounding the lungs and between the lungs. These areas contain skin, muscles, fat, bones, and also the heart and big blood vessels. That translates into a lot of information on the chest radiograph that is not useful for this competition.
# 
# Let's view a normal image from the data set and compare it to the illustration.
# 

# In[ ]:


patientId = df['patientId'][3]
print(patient_class.loc[patientId])

plt.figure(figsize=(10,8))
plt.title("Sample Patient 1 - Normal Image")

draw(parsed[patientId])


# This is an example of a normal chest radiograph (CXR) with ש good technical quality. 
# 
# ## Chest Radiographs Basics  
# [Back to top](#Table-of-Contents)
# 
# In the process of taking the image, an [X-ray](https://en.wikipedia.org/wiki/X-ray) passes through the body and reaches a detector on the other side. Tissues with sparse material, such as lungs which are full of air, do not absorb the X-rays and appear black in the image. Dense tissues such as bones absorb the X-rays and appear white in the image.
# In short - 
# * Black = Air 
# * White = Bone 
# * Grey = Tissue or Fluid
# 
# The left side of the subject is on the right side of the screen by convention. You can also see the small L at the top of the right corner. 
# In a normal image we see the lungs as black, but they have different projections on them - mainly the rib cage bones, main airways, blood vessels and the heart.
# 
# If you are interested in the basic anatomy you can see in a chest radiograph besides the lungs you can view this [5 minutes video by QuickMedic](https://youtu.be/uo7ho8ZW2YY).

# # What are Lung Opacities?
# [Back to top](#Table-of-Contents)
# 
# Let's load an image classified as "Lung Opacity" and see -

# In[ ]:


patientId = df['patientId'][8]
print(patient_class.loc[patientId])

plt.figure(figsize=(10,8))
plt.title("Sample Patient 2 - Lung Opacity")

draw(parsed[patientId])

# ## The Definition of Opacity
# [Back to top](#Table-of-Contents)
# 
# **Opacity** is a pretty loose term - *"Opacity refers to any area that preferentially attenuates the x-ray beam and therefore appears more opaque than the surrounding area. It is a nonspecific term that does not indicate the size or pathologic nature of the abnormality" - from [Felson's Principles of Chest Roentgenology (Fourth Edition)](https://www.amazon.com/Felsons-Principles-Roentgenology-Programmed-Goodman/dp/1455774839?SubscriptionId=AKIAILSHYYTFIVPWUY6Q&tag=duckduckgo-ffab-20&linkCode=xm2&camp=2025&creative=165953&creativeASIN=1455774839).*
# 
# **Meaning** - Any area in the chest radiograph that is more white than it should be. If you compare the images of Sample Patient 1 and Sample Patient 2 you can see that the lower boundry of the lungs of patient 2 is obscured by **opacities**. In the image of Sample Patient 1 you can see the clear difference between the black lungs and the tissue below it, and in the image of Sample Patient 2 there is just this fuzziness. 
# 
# ## What are opacities and what can we understand if we see them?
# *In reply to [Arya Mazaheri](https://www.kaggle.com/aryanet)*
# 
# Usually the lungs are full of air. When someone has **pneumonia**, the air in the lungs is replaced by other material - fluids, bacteria, immune system cells, etc. That's why areas of opacities are areas that are grey but should be more black. When we see them we understand that the lung tissue in that area is probably not healthy. 
# 
# ## Normal vs. Lung Opacity images
# 

# In[ ]:



plt.figure(figsize=(20, 40))

plt.subplot(421)
plt.title("Normal Image")

draw(parsed[df['patientId'][3]])

plt.subplot(423)
draw(parsed[df['patientId'][11]])

plt.subplot(425)
draw(parsed[df['patientId'][12]])

plt.subplot(427)
draw(parsed[df['patientId'][13]])

plt.subplot(422)
plt.title("Lung Opacity")

draw(parsed[df['patientId'][8]])

plt.subplot(424)
draw(parsed[df['patientId'][16]])

plt.subplot(426)
draw(parsed[df['patientId'][19]])

plt.subplot(428)
draw(parsed[df['patientId'][24]])


# In the Lung Opacity images we can see that there is haziness were the labeled boxes are (termed ground glass opacity) and/or a loss of the usual boundries of the lungs (termed consolidation). I will go in depth into these terms in the "A Clear and Detailed Definition of Pneumonia Associated Lung Opacities" section. 
# You can also see that patients with pneumonia are ill and have different cables, stickers, and tubes connected to them. If you see a round white small opacity in and around the lungs it's probably an ECG sticker.
# 
# # Some considerations when building your model
# [Back to top](#Table-of-Contents)
# 
# ## Localization vs. Segmentation of Opacities
# *In reply to [Arya Mazaheri](https://www.kaggle.com/aryanet)*
# 
# As you can see, lung opacities are not homogenoues and they do not have a clear center or clear boundaries. I don't think you can properly segment opacities out of the entire picture because there are no clear boundries. However, if you can segment the lungs properly and filter out the rest of the image, you could create a clean image of the lungs for your neural network to process *(clean as in data cleaning, not clean of pneumonia or opacities)*. An implementation of [U-Net](https://arxiv.org/abs/1505.04597) for such a segmentation can be seen at [Lesson 2: Lung X-Rays Semantic Segmentation](https://colab.research.google.com/github/mdai/ml-lessons/blob/master/lesson2-lung-xrays-segmentation.ipynb) on Google Colab.
# 
# ## How accurate are the annotated boxes?
# *In reply to [Arya Mazaheri](https://www.kaggle.com/aryanet) and [Mikheil Azatov](https://www.kaggle.com/mazatov)*
# 
# There is a known variability between radiologists in the interpretation of chest radiographs. In a study by [Neuman et al. (2012)](https://www.journalofhospitalmedicine.com/jhospmed/article/126918/reliability-cxr-pneumonia), there was only a moderate level of agreement between radiologists about the presence of **infiltrates**, which are opacities by definition.
# Going through the images of this competition, I believe that the presence and position of the annotated boxes would vary between different radiologists.

# # A Closer Look Into "No Lung Opacity / Not Normal" Images
# [Back to top](#Table-of-Contents)

# In[ ]:


patientId = df['patientId'][2]
print(patient_class.loc[patientId])

plt.figure(figsize=(10,8))
plt.title("Sample Patient 3 - Lung Nodules and Masses")
draw(parsed[patientId])

# **There are obvious opacities in this image, so what's going on??**
# 
# This patient has a "Target" value of 0 in the stage_1_train_labels.csv file, which means he does not have a diagnosis of peumonia. *"There is also a binary target column, Target, indicating pneumonia or non-pneumonia" - from the [Data description](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) page*.
# 
# ## Why are there multiple opacities in No Lung Opacity / Not Normal images?
# 
# I'm afraid this competition is misleading in many ways. There are different kinds of opacities (see a general explanation about opacities above). Some are related to pneumonia and some are not. What we see in the image of Patient 3 are Lung Nodules and Masses, which are defined as *"a rounded opacity, well or poorly defined" - [Felson's Principles of Chest Roentgenology (Fourth Edition)](https://www.amazon.com/Felsons-Principles-Roentgenology-Programmed-Goodman/dp/1455774839?SubscriptionId=AKIAILSHYYTFIVPWUY6Q&tag=duckduckgo-ffab-20&linkCode=xm2&camp=2025&creative=165953&creativeASIN=1455774839).* The difference between a **nodule** and a **mass** is the size of the opacity. Unfortunately, it seems like Patient 3 he has multiple lung tumors, probably metastases from an invasive cancer in a different location of the body.
# 
# ## What makes pneumonia associated opacities unique?
# 
# Let's compare "Sample Patient 2 - Lung Opacity", with "Sample Patient 3 - Lung Nodules and Masses"
# 

# In[ ]:


plt.figure(figsize=(20, 40))

plt.subplot(121)
plt.title("Sample Patient 2 - Lung Opacity")
draw(parsed[df['patientId'][8]])

plt.subplot(122)
plt.title("Sample Patient 3 - Lung Nodules and Masses")
draw(parsed[df['patientId'][2]])

# The main difference in the types of opacities between these two patients is the borders and the shape of the opacity, Patient 3 has multiple round and clearly defined opacities. Patient 2 has this poorly defined haziness which obscures the margins of the lungs and heart. This haziness is termed **consolidation**.
# 
# ## Lung opacity annotation instructions to radiologists for this dataset
# [Back to top](#Table-of-Contents)
# 
# [Anouk Stein, MD](https://www.kaggle.com/anoukstein) from MD.ai posted the instructions given to the radiologists who annotated the competition dataset images in [Pneumonia Dataset Annotation Methods](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/discussion/64723#379805):
# 
# > - Lung Opacity (bounding box) - a finding on chest radiograph that in a patient with cough and fever has a high likelihood of being pneumonia
# >
# > - With the understanding that in the absence of clinical information, lateral radiograph, and serial exams, we have to make assumptions
# >
# > - Include any area that is more opaque than the surrounding area (Fleischner definition)
# >
# > - Exclude: obvious mass(es), nodule(s), lobar collapse, linear atelectasis
# >
# > In the cases labeled Not Normal/No Lung Opacity, no lung opacity refers to no opacity suspicious for pneumonia. Other non-pneumonia opacities may be present. Also, some of the not normal cases have subtle abnormalities which require a trained eye to discern. (Which, for now, keeps radiologists around.)
# 
# In the next section I'll try to put it all together and give a clear definition of the opacities annotated in this data set.
# 
# 
# # A Clear and Detailed Definition of Pneumonia Associated Lung Opacities
# [Back to top](#Table-of-Contents)
# 
# Why does the chest radiograph change when a person has pneumonia? To answer this question we have to ask what is pneumonia first.
# 
# ![Pneumonia](https://jamanetwork.com/data/Journals/JAMA/934929/m_jpg160003fa.png)
# Credit: Thompson AE. [Pneumonia. *JAMA.*](https://jamanetwork.com/journals/jama/fullarticle/2488310) 2016;315(6):626. 
# 
# Pneumonia is a lung infection that can be caused by bacteria, viruses, or fungi. Because of the infection and the body's immune response, the sacks in the lungs *(termed alveoli)* are filled with fluids instead of air. The reason that **pneumonia associated lung opacities** look diffuse on the chest radiograph is because the infection and fluid that accumulate spread within the normal tree of airways in the lung. There is no clear border where the infection stops. That is different from other diseases like tumors, which are totally different from the normal lung, and do not maintain the normal structure of the airways inside the lung.
# 
# ## Consolidation vs. Ground-Glass Opacity
# 
# Let's compare two patients -

# In[ ]:


plt.figure(figsize=(20, 40))

plt.subplot(121)
plt.title("Sample Patient 4 - Ground-Glass Opacities")
draw(parsed[df['patientId'][25]])
print(patient_class.loc[df['patientId'][25]])

plt.subplot(122)
plt.title("Sample Patient 5 - Consolidations")
draw(parsed[df['patientId'][28]])
print(patient_class.loc[df['patientId'][28]])

# Patient 4 - **Ground-Glass Opacities**: We can see that the lungs are "whiter" than they should be, but we can see most of the borders of the lungs and heart. 
# **Ground-Glass Opacity** definition - *"On chest radiographs, **ground-glass opacity** appears as an area of hazy increased lung opacity, usually extensive, within which margins of pulmonary vessels may be indistinct.  [...] **Ground-glass opacity** is less opaque than **consolidation**, in which bronchovascular margins are obscured."
# 
# Patient 5 - **Consolidations**: There are fuzzy areas in the lungs and the borders of the lungs and heart cannot be seen.
# **Consolidation** definition -  *"Consolidation appears as a homogeneous increase in pulmonary parenchymal attenuation that obscures the margins of vessels and airway walls."
# 
# These definitions are from [Felson's Principles of Chest Roentgenology (Fourth Edition)](https://www.amazon.com/Felsons-Principles-Roentgenology-Programmed-Goodman/dp/1455774839?SubscriptionId=AKIAILSHYYTFIVPWUY6Q&tag=duckduckgo-ffab-20&linkCode=xm2&camp=2025&creative=165953&creativeASIN=1455774839).
# 
# # Building Your Model for Pneumonia Associated Lung Opacities
# [Back to top](#Table-of-Contents)
# 
# ## Are there pneumonia images in the No Lung Opacity / Not Normal class?
# *In reply to [Yingying.Z](https://www.kaggle.com/yingying6)*
# 
# There should not be any pneumonia cases in the "No Lung Opacity / Not Normal" class because these patients have a "Target" value of 0. *"There is also a binary target column, Target, indicating pneumonia or non-pneumonia"* (from the data description page).
# 
# **However,** the diagnosis of pneumonia is not made only with the chest radiograph so there might be images that look similar to pneumonia. 
# 
# ## How can I predict if the patient has pneumonia?
# I can think of two approaches. The best solution is probably some combination of both of them.
# 1. Try to predict the "Target" from the meta-data, like Ann Antonova's excellent kernel [Practical EDA on numerical data](https://www.kaggle.com/aantonova/practical-eda-on-numerical-data), or Giulia Savorgnan's informative [DATA LEAKAGE: ViewPosition (PA/AP) matters](https://www.kaggle.com/giuliasavorgnan/data-leakage-viewposition-pa-ap-matters).
# 
# 2. I would try to predict the "Target" from the image itself with a neural network, independently of the opacity boxes prediction. 
# 
# 

# # Opacities That Are Not Related to Pneumonia
# 
# ## Pleural effusion
# [Back to top](#Table-of-Contents)

# In[ ]:


plt.figure(figsize=(20, 40))

plt.subplot(121)
plt.title("Sample Patient 6 - Normal")
draw(parsed[df['patientId'][59]])
print(patient_class.loc[df['patientId'][59]])

plt.subplot(122)
plt.title("Sample Patient 7 - Pleural Effusion")
draw(parsed[df['patientId'][125]])
print(patient_class.loc[df['patientId'][125]])

# The lower part of the right lung of Patient 7 (the right lung is at the left side of the picture) is higher than in a normal image. This is a called a **pleural effusion**. It is caused by an accumulation of fluid in the chest outside of the lung. This causes the lung to look smaller on the chest radiograph.
# 
# ## Lung Masses and Nodules
# [Back to top](#Table-of-Contents)

# In[ ]:


plt.figure(figsize=(20, 40))

plt.subplot(121)
plt.title("Sample Patient 6 - Normal")
draw(parsed[df['patientId'][59]])
print(patient_class.loc[df['patientId'][59]])

plt.subplot(122)
plt.title("Sample Patient 3 - Lung Nodules and Masses")
draw(parsed[df['patientId'][2]])
print(patient_class.loc[df['patientId'][2]])

# This is the same image I used in the *A Closer Look Into "No Lung Opacity / Not Normal" Images* section where I explained the difference between a **nodule** and a **pneumonia associated lung opacity**. It's a striking example of lung masses and nodules, I actually don't remember seeing an image with so many clear masses in my work as a physician. Lung Nodules and Masses are defined as *"a rounded opacity, well or poorly defined" - [Felson's Principles of Chest Roentgenology (Fourth Edition)](https://www.amazon.com/Felsons-Principles-Roentgenology-Programmed-Goodman/dp/1455774839?SubscriptionId=AKIAILSHYYTFIVPWUY6Q&tag=duckduckgo-ffab-20&linkCode=xm2&camp=2025&creative=165953&creativeASIN=1455774839).* There are a lot of articles about automated detection of nodules detection in the recent years, you can look at ["Learning to detect chest radiographs containing lung nodules using visual attention networks"](https://arxiv.org/pdf/1712.00996.pdf) as an example.
# 
# ## Increased Vascular Markings + Enlarged Heart
# [Back to top](#Table-of-Contents)

# In[ ]:


plt.figure(figsize=(20, 40))

plt.subplot(121)
plt.title("Sample Patient 6 - Normal")
draw(parsed[df['patientId'][59]])
print(patient_class.loc[df['patientId'][59]])

plt.subplot(122)
plt.title("Sample Patient 8 - Increased Vascular Markings + Enlarged Heart")
draw(parsed[df['patientId'][38]])
print(patient_class.loc[df['patientId'][38]])

# Patient 8 - The heart takes more space than it should for this patient. The heart should not be bigger than half of the chest cavity. Also, there is an increased number and intensity of the lines coming out of the heart. These are pulmonary blood vessels. They are probably congested with blood because the patient's heart is not working properly.
# 
# ## White Lung
# *In reply to [Weiteng](https://www.kaggle.com/weiteng007)*
# 
# [Back to top](#Table-of-Contents)

# In[ ]:


plt.figure(figsize=(15, 15))

plt.subplot(221)
plt.title("Sample Patient 6 - Normal")
draw(parsed[df['patientId'][59]])
print(patient_class.loc[df['patientId'][59]])

plt.subplot(222)
plt.title("Sample Patient 9 - White Lung")
draw(parsed['924f4f8b-fc27-4dfd-b5ae-59c40715e150'])
print(patient_class.loc['924f4f8b-fc27-4dfd-b5ae-59c40715e150'])

plt.subplot(223)
plt.title("Sample Patient 10 - White Lung")
draw(parsed['17a5ce04-809a-42ed-9e58-100cfb33de7a'])
print(patient_class.loc['17a5ce04-809a-42ed-9e58-100cfb33de7a'])

plt.subplot(224)
plt.title("Sample Patient 11 - White Lung")
draw(parsed['9dde630b-1f95-46e6-bcde-117eee4c7283'])
print(patient_class.loc['9dde630b-1f95-46e6-bcde-117eee4c7283'])


# Weiteng007 asked an interesting question - **Can a lung opacity occur if the lung is removed?**. The answer is **yes and no**. An opacity can occur where the lung was once, but it cannot be a **lung opacity**. This **white lung** phenomena in chest radiographs is termed "hemithorax white-out" or "hemithorax opacity". 
# 
# Seeing a "white lung" on a chest radiograph leads to the question - **if we see only an opacity where the lung is supposed to be, what happened to lung?**. These are some possible answers:
# - The lung was removed in a surgery called **pneumonectomy**.
# - The lung is filled with fluid from pneumonia and what we see is a **[pneumonia associated lung opacity](#A-Clear-and-Detailed-Definition-of-Pneumonia-Associated-Lung-Opacities)**. 
# - The lung is there but it is surrounded by fluids inside the chest cavity (termed **[pleural effusion](#Pleural-effusion)**).
# It's hard to tell just by the chest radiograph what is the cause of the "white lung". 
# 
# Since this competition is about penumonia, and these patients have a "Target" value of 1, the cause for their big opacity is most probably pneumonia. Meaning - they have a **pneumonia associated lung opacity** over their entire lung.
# 
# You can see more examples of "white lungs" in [this radiopedia artice](https://radiopaedia.org/articles/hemithorax-white-out-differential).
# 
# ## Unclear Abnormality
# [Back to top](#Table-of-Contents)

# In[ ]:


plt.figure(figsize=(20, 40))

plt.subplot(121)
plt.title("Sample Patient 6 - Normal")
draw(parsed[df['patientId'][59]])
print(patient_class.loc[df['patientId'][59]])

plt.subplot(122)
plt.title("Sample Patient 12 - Unclear Abnormality")
draw(parsed[df['patientId'][40]])
print(patient_class.loc[df['patientId'][40]])

# Patient 12 - I can't see a clear reason for this image to be abnormal. Maybe there are signs of increased vascular markings like Patient 8? I'm not sure. Going through the No Lung Opacity / Not Normal class makes me guess that most of the images in this class are with an unclear abnormality, not something defined. 
# 
# Here is another example -

# In[ ]:


plt.figure(figsize=(20, 40))

plt.subplot(121)
plt.title("Sample Patient 6 - Normal")
draw(parsed[df['patientId'][59]])
print(patient_class.loc[df['patientId'][59]])

plt.subplot(122)
plt.title("Sample Patient 13 - Unclear Abnormality")
draw(parsed[df['patientId'][106]])
print(patient_class.loc[df['patientId'][106]])

# Patient 13 - This is a low quality chest radiograph. The left lung (on the right side of the picture) looks small but that's becuase the image is not symmetrical. The patient is lying down, probably on his side and not flat on his back, and the image was taken at angle and not directly in front of his chest. There are some electrical lines and fluid tubes obscuring parts of the image. However, I cannot find a clear abnormality, excepy maybe increased vascular markings.
# 
# # Summary
# [Back to top](#Table-of-Contents)
# 
# In this **"Pneumonia Detection"** competition you have to do a prediction about **Pneumonia** and about **"Pneumonia Associated Lung Opacities"**. This was a short guide about lung opacities with images from the competition data.
# In my mind, predicting pneumonia ("Target = 1") and prediction of the opacity boxes should be done separately, but I havn't built a model for prediciton yet.
# 
# Good luck everyone!
# 
# Post your questions in the comments section.
