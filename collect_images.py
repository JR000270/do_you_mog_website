import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

class_choice = "not_mog"
#how many images to add on to current
sample_size = 20

cap = cv2.VideoCapture(0)
#for j in range(number_of_classes):
if not os.path.exists(os.path.join(DATA_DIR, class_choice)):
    os.makedirs(os.path.join(DATA_DIR, class_choice))

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
