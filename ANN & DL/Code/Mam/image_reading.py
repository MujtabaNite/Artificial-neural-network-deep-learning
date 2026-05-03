import os
import cv2
import csv

label_dic={'a':0,'b':1,'c':2}
l=list(label_dic.keys())#['a','b','c']
with open(f'alphabets.csv','w',newline='')as csvfile:
  for i in l: #'a', 'b', 'c'
    writer=csv.writer(csvfile)
    path='D:\\FYP-Projects\\Small-Alphabets\\'+i
    for r,d,f in os.walk(path):
      for filename in f:
        if '.png'in filename:
          record=[]
          file_path=os.path.join(path,filename)
          image=cv2.imread(file_path)
          #rest of processing
          label=label_dic[i]#0,1,2
          record.append(label)
          final_image=image.flatten()
          record.extend(final_image)
          writer.writerow(record)
