import os
import cv2

"""
this is for adding in new images to the dataset. it will take a sample of images from the webcam and add them to the specified class folder in data.
command(s): 
cd development
python collect_images.py
"""

DATA_DIR = '../data/test' #replace with testor train the path to data folder. this is where the images will be saved
class_choice = "not_mog" #replace with what folder in data to add to. mog, not mog
if not os.path.exists(DATA_DIR):
    #tell that directory does not exist
    print(f"Directory {DATA_DIR} does not exist. Please create it and try again.")
    exit(1)
if not os.path.exists(os.path.join(DATA_DIR, class_choice)):
    print(f"Directory {DATA_DIR} does not exist. Please create it and try again.")
    exit(1)
    # os.makedirs(os.path.join(DATA_DIR, class_choice))

#how many images to add on to current
sample_size = 20

cap = cv2.VideoCapture(0)


print('Collecting data for class {}'.format(class_choice))

done = False
while True:
    ret, frame = cap.read()
    cv2.putText(frame, 'Ready? Press "Q" to begin sampling', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) == ord('q'):
        break


counter = 0
#need counter + num of files in directory so it doesnt overwrite anything
num_files_in_class_folder = len(os.listdir(os.path.join(DATA_DIR, class_choice)))
while counter < sample_size:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(25)
    cv2.imwrite(os.path.join(DATA_DIR, class_choice, '{}.jpg'.format(counter + num_files_in_class_folder)), frame)
    counter += 1

cap.release()
cv2.destroyAllWindows()
