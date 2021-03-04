# Sprightly-ML

Assumptions:

1.It will only work with barcode and not QR Codes.
2.The bar code is always horizontal in the image of test plate.(A reference attached for the same.)
3.The wells are right above the barcode on plate.

Process:

1.Clone and download the github repository.
3.Make sure python version>3.4 is installed.
4.In terminal type following command:
   pip install -r requirements.txt
4.Run the script by
   python CircleDetection.py <path-of-the-image>
5.An output file namely Cv2HoughCircles.jpg will be saved with Circle detected.  


