import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
def create_output_directory(image_file,path):
    image_name = os.path.splitext(image_file)[0]
    if(path!=save_directory+"\\archive"):
        output_dir = os.path.join(path, image_name)
    else:
        output_dir=path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir
def points(numpoints,max_val):
   MIN_MATCH_COUNT = numpoints
   img1 = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE) # queryImage
   img2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # trainImage
   sift = cv2.SIFT_create()
   kp1, des1 = sift.detectAndCompute(img1,None)
   kp2, des2 = sift.detectAndCompute(img2,None)
   FLANN_INDEX_KDTREE = 1
   index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
   search_params = dict(checks = 50)
   flann = cv2.FlannBasedMatcher(index_params, search_params)
   matches = flann.knnMatch(des1,des2,k=2)
   good = []
   for m,n in matches:
    if m.distance < 0.7*n.distance:
      good.append(m)
   if (len(good)>MIN_MATCH_COUNT) and max_val>-0.08:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
   else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
   draw_params = dict(matchColor = (0,255,0), # draw matches in green color
   singlePointColor = None,
   matchesMask = matchesMask, # draw only inliers
   flags = 2)
   img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
   if (len(good) >= MIN_MATCH_COUNT) and max_val>-0.08 :
        print("The object (e.g., tattoo) exists in both images!")
        #plt.imshow(img3, 'gray')
        #plt.show(block=True)
        create_combined_file()
   else:
     #plt.imshow(img3, 'gray')
     #plt.show(block=True)
         print("The object (e.g., tattoo) does NOT exist or the similarity is too low.")
   return len(good)

# Function to download and save an image
def save_image(url, directory):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # Extract the filename from the URL
        filename = os.path.join(directory, os.path.basename(url))
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
def create_combined_file():
    h, w, _ = template.shape
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    detected_object = template[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    detected_object_resized = cv2.resize(detected_object, (image.shape[1], image.shape[0]))
    combined_image = np.hstack((image, detected_object_resized))
    output_dir = create_output_directory(image_path, save_directory + r"\archive")
    combined_image_path = os.path.join(output_dir, "combined_" + img_url)
    #combined_image_path =  combined_image_path.replace("/", r"\\")
    cv2.imwrite(combined_image_path, combined_image)

# URL of the website to scrape
website_url = 'https://www.ynet.co.il/news/article/r17c5000w6'  # Replace with the website you want to scrape
save_directory = 'C:\python\sites_img'  # Replace with the directory where you want to save the images
template_path = filedialog.askopenfilename(title="Select the Template")
template = cv2.imread(template_path)

# Create the save directory
create_directory(save_directory)

# Send an HTTP GET request to the website
response = requests.get(website_url)

if response.status_code == 200:
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all image tags
    img_tags = soup.find_all('img')
    
    for img in img_tags:
        img_url = img.get('src')
        img_url = urljoin(website_url, img_url)
        save_image(img_url, save_directory)
        image_path = os.path.join(save_directory, os.path.basename(img_url))
        image_path = image_path.replace("/", r"\\")
       # if os.path.isfile(image_path) and any(image_files.lower().endswith(ext) for ext in image_extensions):
        image = cv2.imread(image_path)
        if image is None or template is None:
         print("Error: Unable to load one or both images.")
         exit()
        image_height, image_width, _ = image.shape
        template = cv2.resize(template, (image_width, image_height))
        template = template.astype(image.dtype)
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        threshold = 0.273 
        if max_val >= threshold:
         h, w, _ = template.shape
         top_left = max_loc
         bottom_right = (top_left[0] + w, top_left[1] + h)
         cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
         print("The object (e.g., tattoo) exists in both images!")
         create_combined_file()
        elif(max_val>=0.2):
         points(5,max_val)
           
        
    print("Images downloaded and saved successfully.")
else:
    print(f"Failed to fetch the website. Status code: {response.status_code}")