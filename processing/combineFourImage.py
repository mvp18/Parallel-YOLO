import cv2

for i in range(654):
    # images have indices 4k + 1, 4k + 2, 4k + 3, 4k + 4

    im1 = cv2.imread('resized_' + str(4*i + 1) + '.png', -1)
    im2 = cv2.imread('resized_' + str(4*i + 2) + '.png', -1)
    im3 = cv2.imread('resized_' + str(4*i + 3) + '.png', -1)
    im4 = cv2.imread('resized_' + str(4*i + 4) + '.png', -1)

    hcat1 = cv2.hconcat([im1,im2])
    hcat2 = cv2.hconcat([im3,im4])

    vcat1 = cv2.vconcat([hcat1,hcat2])

    final = cv2.resize(vcat1, (416, 416))

    cv2.imwrite("combined_" + str(i + 1) + ".png", final)
