import cv2
import numpy as np
from model import CANNet
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from matplotlib import cm
# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('.\\videos\\crowd2.mp4')

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video  file")
model = CANNet()

model = model.cuda()
checkpoint = torch.load('_model_best.pth.tar')

model.load_state_dict(checkpoint['state_dict'])
model.eval()
transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
])
# Read until video is completed
i = 0
def plotDensity(density):
    '''
    @density: np array of corresponding density map
    @plot_path: path to save the plot
    '''
    density = density * 255.0
    #print("SHAPE: ",density.shape)
    # plot with overlay
    colormap_i = cm.jet(density)[:, :, 0:3]

    overlay_i = colormap_i

    new_map = overlay_i.copy()
    new_map[:, :, 0] = overlay_i[:, :, 2]
    new_map[:, :, 2] = overlay_i[:, :, 0]
    #print(new_map)
    # cv2.imshow('xx', new_map)
    # cv2.waitKey(1000)
    # cv2.imwrite(plot_path, new_map * 255)
    return new_map * 255
def writeCountNumber(number, image):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (50, 50)

    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.putText() method
    image = cv2.putText(image, f"Persons: {number}", org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    return image
den_map = None
crow_count = 0
while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # Display the resulting frame
        frame = cv2.resize(frame, (512, 384))

        if i%30 == 0:
            img = transform(frame).cuda()
            img = img.unsqueeze(0)
            img = Variable(img.cuda())
            density = model(img).data.cpu().numpy()
            # print(density.sum())
            den_map = plotDensity(np.asarray(density[0][0]))
            den_map = cv2.resize(den_map, (512, 384))
            cv2.imshow('Density', den_map)
            crow_count = density.sum()
        i+=1
        # if den_map is not None:
        #     vis = cv2.vconcat([frame,den_map])
        frame = writeCountNumber(crow_count, frame)
        cv2.imshow('Frame', frame)



        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()