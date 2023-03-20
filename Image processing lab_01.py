#!/usr/bin/env python
# coding: utf-8

# In[6]:

#eathgteg
import tarfile
import os
import numpy as np

# allahabas
data_base_path = os.path.join('data')
print(data_base_path)
data_folder = 'lab-01-data'
tar_path = os.path.join(data_base_path, data_folder + '.tar.gz')

if not os.path.exists(tar_path):
    raise Exception('Path to is not valid {}'.format(tar_path))
    
with tarfile.open(tar_path, mode='r:gz') as tar:
    tar.extractall(path=data_base_path)


# In[7]:


import skimage.io
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

# Load images
path_to_image = os.path.join(data_base_path, data_folder, "mucinous_adeno.jpg")
path_to_gt = os.path.join(data_base_path, data_folder, "mucinous_adeno_gt.jpg")
adeno_img = skimage.io.imread(path_to_image)
adeno_gt = skimage.io.imread(path_to_gt) > 128

# Get size of the image in px
im_h, im_w, im_c = adeno_img.shape

# Display MRI image
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot image
ax[0].imshow(adeno_img, interpolation='nearest')
ax[0].set_title('H&E image'.format(im_h, im_w))
ax[0].axis('off')

# Plot image with overlay
ax[1].imshow(adeno_img, interpolation='nearest')
ax[1].imshow(adeno_gt, alpha=0.5, interpolation='nearest')
ax[1].set_title('Annotation of mucinous area (yellow)')
ax[1].axis('off')
plt.suptitle("Mucinous adenocarcinoma\n({} px, {} px)".format(im_h, im_w))
plt.show()


# In[35]:


fig, axs = plt.subplots(ncols=3, figsize=(15,5))
axs[0].imshow(adeno_img[:,:,0], interpolation='nearest')
axs[1].imshow(adeno_img[:,:,1], interpolation='nearest')
axs[2].imshow(adeno_img[:,:,2], interpolation='nearest')
fig.show()


# In[124]:


r = adeno_img[:,:,0].ravel()
g = adeno_img[:,:,1].ravel()
b = adeno_img[:,:,2].ravel()
gb = np.dstack((g,b))

n_bins = 256
fig, axs = plt.subplots(ncols=3, figsize=(15,5))
axs[0].hist(r,bins=n_bins)
axs[1].hist(g,bins=n_bins)
axs[2].hist(b,bins=n_bins)
fig.show()


# In[65]:


import sklearn
from sklearn.mixture import GaussianMixture
gm_red = GaussianMixture(n_components=2).fit(r.reshape(-1,1))
gm_green = GaussianMixture(n_components=2).fit(g.reshape(-1,1))
gm_blue = GaussianMixture(n_components=2).fit(b.reshape(-1,1))

#gm_clt.means_
r_binary = np.array([gm_red.means_[label] for label in gm_red.predict(r.reshape(-1,1))])
g_binary = np.array([gm_green.means_[label] for label in gm_green.predict(g.reshape(-1,1))])
b_binary = np.array([gm_blue.means_[label] for label in gm_blue.predict(b.reshape(-1,1))])


# In[98]:


r_mean = np.ones(adeno_img.shape[:2])*np.mean(adeno_img[:,:,0].ravel())


# In[76]:


#print(g.reshape(-1,1).reshape(1,-1))
#print(g_binary.reshape(1,-1)[0].reshape(im_h, im_w))
r_binary = r_binary.reshape(1,-1)[0].reshape(im_h, im_w)
g_binary = g_binary.reshape(1,-1)[0].reshape(im_h, im_w)
b_binary = b_binary.reshape(1,-1)[0].reshape(im_h, im_w)


# In[117]:


adeno_pred1 = np.array([r_mean,g_binary,b_binary],dtype=float)
rgb = np.dstack((r_mean,g_binary,b_binary)).astype(np.uint8)
#r.reshape(im_h,im_w).shape
lst = np.dstack((g_binary.ravel(),b_binary.ravel()))
lst = np.array([x[0]+x[1] for x in lst[0]])
np.unique(lst)
plt.hist(lst)


# In[131]:


fig, axs = plt.subplots(ncols=3, figsize=(15,5))
axs[0].imshow(adeno_img[:,:,1], interpolation='nearest')
axs[1].imshow(g_binary, interpolation='nearest')
axs[2].imshow(b_binary, interpolation='nearest')


# In[101]:


fig, ax = plt.subplots(1, 3, figsize=(12, 6))

# Plot image
ax[0].imshow(adeno_img, interpolation='nearest')
ax[0].set_title('H&E image'.format(im_h, im_w))
ax[0].axis('off')

# Plot image with overlay
ax[1].imshow(rgb, interpolation='nearest')
ax[2].imshow(adeno_gt, alpha=0.5, interpolation='nearest')
ax[2].set_title('Annotation of mucinous area (yellow)')
#ax[1].imshow(adeno_gt, alpha=0.5, interpolation='nearest')
#ax[1].set_title('Annotation of mucinous area (yellow)')
#ax[1].axis('off')
plt.suptitle("Mucinous adenocarcinoma\n({} px, {} px)".format(im_h, im_w))
plt.show()


# In[144]:


g_binary = (g_binary > 150)
error = 1 - np.sum((adeno_gt==g_binary))/(adeno_gt.shape[0]*adeno_gt.shape[1])
print(error)


# Closing

# In[167]:


from skimage import morphology
fig, axs = plt.subplots(ncols=4,nrows=2,figsize = (16,5))

Radius = [1,2,3,5]
for i in range(4):
    closing = morphology.binary_closing(g_binary,morphology.disk(radius=Radius[i]))
    axs[0][i].imshow(g_binary, interpolation='nearest')
    axs[1][i].imshow(closing, interpolation='nearest')
fig.show()
fig.tight_layout()


# Opening

# In[172]:


fig, axs = plt.subplots(ncols=4,nrows=2,figsize = (18,8))

closing = morphology.binary_closing(g_binary,morphology.disk(radius=2))
Radius = [1,2,3,5]
for i in range(4):
    opening = morphology.binary_opening(closing,morphology.disk(radius=Radius[i]))
    axs[0][i].imshow(closing, interpolation='nearest')
    axs[1][i].imshow(opening, interpolation='nearest')
fig.show()
fig.tight_layout()


# Holes

# In[176]:


from skimage import morphology
fig, axs = plt.subplots(ncols=4,nrows=2,figsize = (16,8))

opening = morphology.binary_opening(closing,morphology.disk(radius=5))
thresh = [10,50,100,500]
for i in range(4):
    no_holes = morphology.remove_small_holes(opening,area_threshold=thresh[i])
    axs[0][i].imshow(opening, interpolation='nearest')
    axs[1][i].imshow(no_holes, interpolation='nearest')
fig.show()
fig.tight_layout()

