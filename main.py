# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
'''import cv2
import os'''

import hog
import lbp
import hough
import output


'''def load_images_from_folder(folder):
    images = []
    count = 0
    for filename in os.listdir(folder):
        count = count+1
        if(count>10):
            break
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_COLOR)

        if img is not None:
            # Creating GUI window to display an image on screen
            # first Parameter is windows title (should be in string format)
            # Second Parameter is image array
            #cv2.imshow("image", img)

            # To hold the window on screen, we use cv2.waitKey method
            # Once it detected the close input, it will release the control
            # To the next line
            # First Parameter is for holding screen for specified milliseconds
            # It should be positive integer. If 0 pass an parameter, then it will
            # hold the screen until user close it.
            #cv2.waitKey(0)

            # It is for removing/deleting created GUI window from screen
            # and memory
            #cv2.destroyAllWindows()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            cv2.imshow('Original image', img)
            cv2.imshow('Gray image', gray)

            cv2.waitKey(0)
            cv2.destroyAllWindows()


            images.append(img)
    return images'''

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #boot_images = load_images_from_folder("DataSet/Boot")
    #sandal_images = load_images_from_folder("DataSet/Sandal")
    #shoe_images = load_images_from_folder("DataSet/Shoe")

    #detect_line_using_hough(boot_images)

    #get_hog(boot_images)
    #getLBP(images=boot_images)
    lbp.get_lbp_prediction()
    #hog.get_hog_prediction()
    #hough.get_hough_result()
    #output.get_lbp_reports()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
